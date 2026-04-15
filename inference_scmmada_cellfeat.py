import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer

from models import MMadaConfig, MMadaModelLM
from training.prompting_utils import UniversalPrompting, reserved_token_mapping


FEATURE_FILES = [
    'okrcell_sft_features_archs4.h5ad',
    'okrcell_sft_features_census.h5ad',
]


def clean_question_text(text: str) -> str:
    return str(text).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '').strip()


def build_question_prompt(question: str) -> str:
    question = clean_question_text(question)
    return (
        '<|start_header_id|>user<|end_header_id|>\n\n'
        f'{question}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    )


def resolve_tokenizer_path(cfg) -> str:
    return cfg.model.mmada.get('tokenizer_path', cfg.model.mmada.pretrained_model_path)


def load_model_and_tokenizer(cfg_path: str, checkpoint_path: str, device: torch.device):
    cfg = OmegaConf.load(cfg_path)
    tokenizer_path = resolve_tokenizer_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left')

    base_config = AutoConfig.from_pretrained(tokenizer_path).to_dict()
    mmada_config_dict = {k: v for k, v in cfg.model.mmada.items()}
    mmada_config_dict['pretrained_model_path'] = checkpoint_path
    merged_config = {**base_config, **mmada_config_dict}
    mmada_config = MMadaConfig(**merged_config)
    mmada_config.new_vocab_size = int(max(int(getattr(mmada_config, 'llm_vocab_size', 0)), len(tokenizer), max(reserved_token_mapping.values()) + 1))

    model = MMadaModelLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16, config=mmada_config)
    model.resize_token_embeddings(mmada_config.new_vocab_size)
    model.config.embedding_size = model.config.vocab_size
    model = model.to(device)

    model.init_gene_expression_value_encoder(
        hidden_dim=cfg.training.get('gene_expression_hidden_dim', None),
        dropout=float(cfg.training.get('gene_expression_dropout', 0.0)),
        max_value=float(cfg.training.get('gene_expression_max_value', 20.0)),
    )
    if cfg.dataset.params.get('cell_feature_root', None):
        model.init_cell_feature_soft_tokenizer(
            input_dim=int(cfg.dataset.params.get('cell_feature_dim', 768)),
            num_soft_tokens=int(cfg.training.get('cell_feature_num_soft_tokens', 4)),
            hidden_dim=cfg.training.get('cell_feature_hidden_dim', None),
            dropout=float(cfg.training.get('cell_feature_dropout', 0.0)),
        )

    # Reload full checkpoint after conditioning modules are initialized.
    bin_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
    bin_index = os.path.join(checkpoint_path, 'pytorch_model.bin.index.json')
    safe_path = os.path.join(checkpoint_path, 'model.safetensors')
    safe_index = os.path.join(checkpoint_path, 'model.safetensors.index.json')
    if os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    elif os.path.exists(bin_index) or os.path.exists(safe_index):
        from transformers.modeling_utils import load_sharded_checkpoint
        load_sharded_checkpoint(model, checkpoint_path, strict=False)
    elif os.path.exists(safe_path):
        from safetensors.torch import load_file
        state_dict = load_file(safe_path)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f'No checkpoint weights found under {checkpoint_path}')

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=cfg.dataset.preprocessing.max_seq_length,
        special_tokens=(
            '<|soi|>', '<|eoi|>', '<|sov|>', '<|eov|>', '<|t2i|>',
            '<|mmug|>', '<|mmu|>', '<|t2v|>', '<|v2v|>', '<|lvg|>'
        ),
        ignore_id=-100,
        cond_dropout_prob=cfg.training.cond_dropout_prob,
        use_reserved_token=True,
    )
    model.eval()
    return cfg, tokenizer, uni_prompting, model


def find_cell_feature(cell_id: str, feature_root: str) -> np.ndarray:
    for filename in FEATURE_FILES:
        path = os.path.join(feature_root, filename)
        if not os.path.exists(path):
            continue
        handle = ad.read_h5ad(path, backed='r')
        try:
            ids = handle.obs['cell_id'].astype(str).tolist() if 'cell_id' in handle.obs.columns else [str(x) for x in handle.obs_names]
            for idx, current_id in enumerate(ids):
                if current_id == cell_id:
                    return np.asarray(handle.X[idx]).reshape(-1)
        finally:
            handle.file.close()
    raise KeyError(f'cell_id not found in feature files: {cell_id}')


@torch.no_grad()
def generate_answer(
    model: MMadaModelLM,
    tokenizer,
    uni_prompting: UniversalPrompting,
    cell_features: torch.Tensor,
    question: str,
    max_new_tokens: int = 64,
    steps: int = 64,
    block_length: int = 64,
    temperature: float = 0.0,
):
    prompt_text = build_question_prompt(question)
    gene_ids = torch.zeros((1, 0), dtype=torch.long, device=cell_features.device)
    input_ids, _ = uni_prompting((gene_ids, [prompt_text]), 'mmug_gen')
    input_ids = input_ids.to(cell_features.device)

    mask_id = model.config.mask_token_id
    x = torch.full((1, input_ids.shape[1] + max_new_tokens), mask_id, dtype=torch.long, device=cell_features.device)
    x[:, :input_ids.shape[1]] = input_ids
    prompt_index = x != mask_id

    assert max_new_tokens % block_length == 0
    num_blocks = max_new_tokens // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    prefix_length = model.cell_feature_soft_tokenizer.num_soft_tokens if model.cell_feature_soft_tokenizer is not None else 0

    for num_block in range(num_blocks):
        for _ in range(steps_per_block):
            mask_index = x == mask_id
            attention_bias = torch.ones(x.shape[0], 1, x.shape[1], x.shape[1], device=x.device)
            if prefix_length > 0:
                prefix_ids = torch.full((x.shape[0], prefix_length), tokenizer.pad_token_id or tokenizer.eos_token_id, dtype=x.dtype, device=x.device)
                input_ids_for_model = torch.cat([prefix_ids, x], dim=1)
                inputs_embeds = model.build_inputs_embeds_with_conditioning(input_ids=x, cell_features=cell_features, gene_token_start=2)
                attention_bias = model.extend_attention_bias_for_prefix(attention_bias, prefix_length)
                logits = model(input_ids=input_ids_for_model, inputs_embeds=inputs_embeds, attention_bias=attention_bias).logits
                logits = logits[:, prefix_length:, :]
            else:
                logits = model(input_ids=x, attention_bias=attention_bias).logits

            if temperature > 0:
                noise = -torch.log(-torch.log(torch.rand_like(logits, dtype=torch.float32).clamp_min(1e-12))).to(logits.dtype)
                logits = logits + temperature * noise

            x0 = torch.argmax(logits, dim=-1)
            p = F.softmax(logits.float(), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, input_ids.shape[1] + (num_block + 1) * block_length:] = -float('inf')

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float('inf')))

            block_mask = x[:, input_ids.shape[1] + num_block * block_length: input_ids.shape[1] + (num_block + 1) * block_length] == mask_id
            remaining = int(block_mask.sum().item())
            if remaining == 0:
                continue
            transfer_k = max(1, remaining // steps_per_block)
            transfer_index = torch.zeros_like(x, dtype=torch.bool)
            _, select_index = torch.topk(confidence[0], k=min(transfer_k, remaining))
            transfer_index[0, select_index] = True
            x[transfer_index] = x0[transfer_index]

    generated_ids = x[0, input_ids.shape[1]:].tolist()
    stop_tokens = {
        tokenizer.eos_token_id,
        int(uni_prompting.sptids_dict['<|eot_id|>']),
        reserved_token_mapping['[iPAD]'],
    }
    trimmed = []
    for token_id in generated_ids:
        if token_id in stop_tokens:
            break
        trimmed.append(token_id)
    return tokenizer.decode(trimmed, skip_special_tokens=True).strip(), prompt_text, input_ids[0].tolist(), generated_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cell-id', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--feature-root', default=None)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--block-length', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--output-json', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg, tokenizer, uni_prompting, model = load_model_and_tokenizer(args.config, args.checkpoint, device)
    feature_root = args.feature_root or cfg.dataset.params.cell_feature_root
    feature = find_cell_feature(args.cell_id, feature_root)
    cell_features = torch.tensor(feature, dtype=torch.float32, device=device).unsqueeze(0)

    answer, prompt_text, prompt_ids, generated_ids = generate_answer(
        model=model,
        tokenizer=tokenizer,
        uni_prompting=uni_prompting,
        cell_features=cell_features,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        block_length=args.block_length,
        temperature=args.temperature,
    )

    result = {
        'cell_id': args.cell_id,
        'question': args.question,
        'prompt_text': prompt_text,
        'predicted_answer': answer,
        'prompt_ids': prompt_ids,
        'generated_ids': generated_ids,
    }

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
