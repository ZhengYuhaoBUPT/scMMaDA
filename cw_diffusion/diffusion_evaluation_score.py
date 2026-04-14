import argparse
import copy
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_utils import load_sharded_checkpoint

from models import MMadaConfig, MMadaModelLM
from training.data import CellFeatureConversationDataset, load_gene_vocab
from training.prompting_utils import UniversalPrompting, reserved_token_mapping

LOG = logging.getLogger("cw_diffusion")


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion matched-vs-mismatched evaluation for scMMaDA.")
    parser.add_argument("--train-config", required=True, help="Training config yaml used for the model.")
    parser.add_argument("--model-path", required=True, help="Checkpoint path to load for evaluation. Prefer checkpoint-xxx/unwrapped_model.")
    parser.add_argument("--evaluation-dataset", required=True, help="Conversation json file with id + conversations.")
    parser.add_argument("--feature-root", required=True, help="Directory or file containing .h5ad cell features.")
    parser.add_argument("--output-csv", required=True, help="Where to write the matched/mismatched scoring table.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-negatives", type=int, default=10)
    parser.add_argument("--score-repeats", type=int, default=4, help="Average multiple random mask draws per pairing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--shuffle-mode", default="transcriptome", choices=["transcriptome", "llm-response"])
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for quick evaluation.")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


class DiffusionEvalDataset(Dataset):
    def __init__(self, conversation_json_path: str, cell_feature_root: str, tokenizer, max_samples: int = 0):
        dummy_cfg = type("DummyCfg", (), {})()
        dataset = CellFeatureConversationDataset(
            conversation_json_path=conversation_json_path,
            cell_feature_root=cell_feature_root,
            tokenizer=tokenizer,
            max_seq_length=256,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
        )
        self.samples = dataset.samples[: max_samples or None]
        self.feature_index = dataset.feature_index
        self.feature_paths = dataset.feature_paths
        self.feature_handles = dataset.feature_handles
        self._get_feature_handle = dataset._get_feature_handle
        self._format_conversation_text = dataset._format_conversation_text
        self._clean_message_text = dataset._clean_message_text

    def __len__(self):
        return len(self.samples)

    def _load_feature(self, sample_id: str) -> torch.Tensor:
        feature_path, row_idx = self.feature_index[sample_id]
        handle = self._get_feature_handle(feature_path)
        vec = np.asarray(handle.X[row_idx]).reshape(-1)
        return torch.tensor(vec, dtype=torch.float32)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        sample_id = str(item.get("id") or item.get("image"))
        text = self._format_conversation_text(item.get("conversations", []))
        response = ""
        for turn in reversed(item.get("conversations", [])):
            if isinstance(turn, dict) and turn.get("from") == "gpt":
                response = self._clean_message_text(turn.get("value", ""))
                break
        return {
            "sample_id": sample_id,
            "texts": text,
            "response": response,
            "cell_features": self._load_feature(sample_id),
            "gene_ids": torch.empty((0,), dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: List[Dict]):
        return {
            "sample_ids": [x["sample_id"] for x in batch],
            "texts": [x["texts"] for x in batch],
            "responses": [x["response"] for x in batch],
            "cell_features": torch.stack([x["cell_features"] for x in batch]),
            "gene_ids": torch.stack([x["gene_ids"] for x in batch]),
        }


def build_model(train_cfg, model_path: str, dtype: torch.dtype, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model.mmada.pretrained_model_path, padding_side="left")

    base_config = AutoConfig.from_pretrained(train_cfg.model.mmada.pretrained_model_path).to_dict()
    mmada_config_dict = {k: v for k, v in train_cfg.model.mmada.items()}
    merged_config = {**base_config, **mmada_config_dict}
    mmada_config = MMadaConfig(**merged_config)

    text_vocab_size = max(int(getattr(mmada_config, "llm_vocab_size", 0)), len(tokenizer))
    base_vocab_size = max(text_vocab_size, max(reserved_token_mapping.values()) + 1)
    gene_token_offset = int(base_vocab_size)
    gene_vocab_num_embeddings = 0
    gene_vocab_path = train_cfg.dataset.params.get("gene_vocab_path", None)
    if gene_vocab_path is not None and os.path.exists(gene_vocab_path):
        _, _, gene_vocab_num_embeddings, _ = load_gene_vocab(gene_vocab_path)
    mmada_config.new_vocab_size = int(gene_token_offset + gene_vocab_num_embeddings)

    model = MMadaModelLM.from_pretrained(
        train_cfg.model.mmada.pretrained_model_path,
        torch_dtype=dtype,
        config=mmada_config,
    )
    model.resize_token_embeddings(mmada_config.new_vocab_size)
    model.config.embedding_size = model.config.vocab_size
    model.init_gene_expression_value_encoder(
        hidden_dim=train_cfg.training.get("gene_expression_hidden_dim", None),
        dropout=float(train_cfg.training.get("gene_expression_dropout", 0.0)),
        max_value=float(train_cfg.training.get("gene_expression_max_value", 20.0)),
    )
    model.init_cell_feature_soft_tokenizer(
        input_dim=int(train_cfg.dataset.params.get("cell_feature_dim", 768)),
        num_soft_tokens=int(train_cfg.training.get("cell_feature_num_soft_tokens", 4)),
        hidden_dim=train_cfg.training.get("cell_feature_hidden_dim", None),
        dropout=float(train_cfg.training.get("cell_feature_dropout", 0.0)),
    )

    if os.path.isdir(model_path):
        if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            del state_dict
        elif os.path.exists(os.path.join(model_path, "model.safetensors")):
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(model_path, "model.safetensors"))
            model.load_state_dict(state_dict, strict=False)
            del state_dict
        elif os.path.exists(os.path.join(model_path, "pytorch_model.bin.index.json")) or os.path.exists(os.path.join(model_path, "model.safetensors.index.json")):
            load_sharded_checkpoint(model, model_path, strict=False)
        else:
            raise FileNotFoundError(f"Unsupported checkpoint layout at {model_path}")
    else:
        raise FileNotFoundError(f"Model path not found: {model_path}")

    model.to(device)
    model.eval()
    return tokenizer, model


def build_prompting(tokenizer, train_cfg):
    return UniversalPrompting(
        tokenizer,
        max_text_len=train_cfg.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
            "<|mmug|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=0.0,
        use_reserved_token=True,
    )


def prepare_inputs_and_labels_for_mmu(input_ids_mmu, prompt_masks, labels_mmu, mask_id, eps=1e-3):
    b, l = input_ids_mmu.shape
    t = torch.rand(b, device=input_ids_mmu.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    prompt_masks_bool = prompt_masks.bool()
    target_masks = ~prompt_masks_bool
    masked_indices = (torch.rand((b, l), device=input_ids_mmu.device) < p_mask) & target_masks

    for i in range(b):
        if masked_indices[i].any():
            continue
        valid_positions = torch.nonzero(target_masks[i], as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            raise ValueError("MMUG sample has no non-prompt target positions to mask.")
        masked_indices[i, valid_positions[0]] = True

    noisy_batch = torch.where(masked_indices, mask_id, input_ids_mmu)
    prompt_masks = prompt_masks.to(torch.int64)
    answer_lengths = torch.sum((1 - prompt_masks), dim=-1, keepdim=True)
    answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])

    labels_masked = labels_mmu.clone()
    labels_masked[~masked_indices] = -100
    return noisy_batch, labels_masked, p_mask, answer_lengths, masked_indices


def compute_masked_diffusion_loss_per_example(logits, input_ids_masked, labels_masked, p_mask_masked, answer_lengths_masked, mask_id):
    masked_indices = input_ids_masked == mask_id
    batch_size = logits.shape[0]
    if not masked_indices.any():
        return torch.zeros(batch_size, device=logits.device)

    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels_masked.view(-1),
        ignore_index=-100,
        reduction='none',
    ).view(batch_size, -1)
    weights = torch.zeros_like(ce)
    weights[masked_indices] = 1.0 / p_mask_masked[masked_indices]
    per_token = ce * weights
    per_token = torch.where(masked_indices, per_token / answer_lengths_masked, torch.zeros_like(per_token))
    return per_token.sum(dim=1)


def make_derangement(n: int, seed: int) -> np.ndarray:
    if n <= 1:
        raise ValueError("Need at least two samples to build mismatched transcriptome pairs.")
    rng = np.random.default_rng(seed)
    perm = np.arange(n)
    while True:
        rng.shuffle(perm)
        if np.all(perm != np.arange(n)):
            return perm


def score_batch(model, tokenizer, uni_prompting, batch, train_cfg, device, score_repeats: int):
    gene_ids = batch["gene_ids"].to(device, non_blocking=True)
    texts = batch["texts"]
    cell_features = batch["cell_features"].to(device, non_blocking=True)
    mask_id = model.config.mask_token_id
    scores = []

    for rep in range(score_repeats):
        input_ids_mmug, prompt_masks_mmug, labels_mmug = uni_prompting((texts, gene_ids), 'mmug')
        input_ids_mmug = input_ids_mmug.to(device, non_blocking=True)
        labels_mmug = labels_mmug.to(device, non_blocking=True)
        prompt_masks_mmug = prompt_masks_mmug.to(device, non_blocking=True)

        input_ids_mmug, labels_mmug, p_mask_mmug, answer_lengths_mmug, _ = prepare_inputs_and_labels_for_mmu(
            input_ids_mmug, prompt_masks_mmug, labels_mmug, mask_id
        )

        attention_bias_mmug = torch.ones(
            input_ids_mmug.shape[0], 1, input_ids_mmug.shape[1], input_ids_mmug.shape[1],
            device=input_ids_mmug.device,
        )

        prefix_length = model.cell_feature_soft_tokenizer.num_soft_tokens
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        prefix_ids = torch.full(
            (input_ids_mmug.shape[0], prefix_length),
            pad_token_id,
            dtype=input_ids_mmug.dtype,
            device=input_ids_mmug.device,
        )
        input_ids_for_loss = torch.cat([prefix_ids, input_ids_mmug], dim=1)
        labels_for_loss = torch.cat([torch.full_like(prefix_ids, -100), labels_mmug], dim=1)
        p_mask_for_loss = torch.cat([torch.ones_like(prefix_ids, dtype=p_mask_mmug.dtype), p_mask_mmug], dim=1)
        answer_lengths_for_loss = torch.cat([
            torch.ones_like(prefix_ids, dtype=answer_lengths_mmug.dtype), answer_lengths_mmug
        ], dim=1)
        attention_bias_mmug = model.extend_attention_bias_for_prefix(attention_bias_mmug, prefix_length)
        inputs_embeds_mmug = model.build_inputs_embeds_with_conditioning(
            input_ids=input_ids_mmug,
            cell_features=cell_features,
            gene_expression=None,
            gene_token_start=2,
        )

        with torch.autocast(device_type='cuda', dtype=next(model.parameters()).dtype, enabled=(device.type == 'cuda')):
            logits = model(
                input_ids=input_ids_for_loss,
                inputs_embeds=inputs_embeds_mmug,
                attention_bias=attention_bias_mmug,
            ).logits
        scores.append(
            compute_masked_diffusion_loss_per_example(
                logits, input_ids_for_loss, labels_for_loss, p_mask_for_loss, answer_lengths_for_loss, mask_id
            ).detach().float().cpu()
        )

    return torch.stack(scores, dim=0).mean(dim=0)


def evaluate_dataset(model, tokenizer, uni_prompting, dataset, train_cfg, args):
    device = torch.device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    all_rows = []

    LOG.info("Scoring matched pairs")
    for batch_idx, batch in enumerate(loader):
        matched_scores = score_batch(model, tokenizer, uni_prompting, batch, train_cfg, device, args.score_repeats)
        for i, score in enumerate(matched_scores.tolist()):
            all_rows.append({
                "question_id": len(all_rows),
                "sample_id": batch["sample_ids"][i],
                "type": "correct",
                "response": batch["responses"][i],
                "eval_all_perplexities": float(score),
                "eval_all_scores": float(score),
                "replicate": 0,
            })
        if (batch_idx + 1) % 10 == 0:
            LOG.info("Matched batches: %d/%d", batch_idx + 1, len(loader))

    base_question_ids = [row["question_id"] for row in all_rows]
    sample_ids = [dataset[idx]["sample_id"] for idx in range(len(dataset))]
    responses = [dataset[idx]["response"] for idx in range(len(dataset))]
    texts = [dataset[idx]["texts"] for idx in range(len(dataset))]
    features = torch.stack([dataset[idx]["cell_features"] for idx in range(len(dataset))])
    gene_ids = torch.stack([dataset[idx]["gene_ids"] for idx in range(len(dataset))])

    LOG.info("Scoring mismatched pairs with mode=%s", args.shuffle_mode)
    for rep in range(args.num_negatives):
        if args.shuffle_mode == "transcriptome":
            perm = make_derangement(len(dataset), args.seed + rep)
            shuffled_features = features[perm]
            shuffled_texts = texts
            shuffled_responses = responses
        else:
            perm = make_derangement(len(dataset), args.seed + rep)
            shuffled_features = features
            shuffled_texts = []
            shuffled_responses = []
            for i in range(len(dataset)):
                item = copy.deepcopy(dataset.samples[i])
                item["conversations"][-1]["value"] = dataset._clean_message_text(dataset.samples[int(perm[i])]["conversations"][-1]["value"])
                shuffled_texts.append(dataset._format_conversation_text(item["conversations"]))
                shuffled_responses.append(item["conversations"][-1]["value"])

        rep_loader = DataLoader(
            list(range(len(dataset))),
            batch_size=args.batch_size,
            shuffle=False,
        )
        offset = 0
        for batch_ids in rep_loader:
            batch_ids = batch_ids.tolist()
            batch = {
                "sample_ids": [sample_ids[i] for i in batch_ids],
                "texts": [shuffled_texts[i] for i in batch_ids],
                "responses": [shuffled_responses[i] if args.shuffle_mode == "llm-response" else responses[i] for i in batch_ids],
                "cell_features": shuffled_features[batch_ids],
                "gene_ids": gene_ids[batch_ids],
            }
            incorrect_scores = score_batch(model, tokenizer, uni_prompting, batch, train_cfg, device, args.score_repeats)
            for local_i, score in enumerate(incorrect_scores.tolist()):
                sample_idx = batch_ids[local_i]
                all_rows.append({
                    "question_id": base_question_ids[sample_idx],
                    "sample_id": sample_ids[sample_idx],
                    "type": "incorrect",
                    "response": batch["responses"][local_i],
                    "eval_all_perplexities": float(score),
                    "eval_all_scores": float(score),
                    "replicate": rep,
                })
            offset += len(batch_ids)
        LOG.info("Finished negative replicate %d/%d", rep + 1, args.num_negatives)

    return pd.DataFrame(all_rows)


def main():
    args = parse_args()
    setup_logging()
    seed_everything(args.seed)

    train_cfg = OmegaConf.load(args.train_config)
    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)

    tokenizer, model = build_model(train_cfg, args.model_path, dtype, device)
    dataset = DiffusionEvalDataset(args.evaluation_dataset, args.feature_root, tokenizer, max_samples=args.max_samples)
    uni_prompting = build_prompting(tokenizer, train_cfg)

    LOG.info("Loaded evaluation dataset with %d samples", len(dataset))
    df = evaluate_dataset(model, tokenizer, uni_prompting, dataset, train_cfg, args)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOG.info("Saved diffusion evaluation table to %s", output_path)


if __name__ == "__main__":
    main()
