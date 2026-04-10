# Copyright 2025 MMaDA Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import itertools
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from training.data import Text2ImageDataset, CellwTextDataset, load_gene_vocab
from training.utils import get_config, flatten_omega_conf, image_transform
from training.imagenet_dataset import ImageNetDataset
from parquet import RefinedWebDataset

from models import MAGVITv2, get_mask_schedule, MMadaModelLM, MMadaConfig
from training.prompting_utils import UniversalPrompting, reserved_token_mapping
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


SYSTEM_PROMPT_LEN = 28

from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == "vq16":
        return VQ_16
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    mmug_only = bool(config.training.get("mmug_only", False))
    batch_size_mmug_cfg = (
        config.training.batch_size_mmug if hasattr(config.training, "batch_size_mmug")
        else (
            config.training.batch_size_g2t if hasattr(config.training, "batch_size_g2t")
            else config.training.batch_size_t2i
        )
    )
    batch_size_t2g_cfg = int(config.training.get("batch_size_t2g", batch_size_mmug_cfg))

    if mmug_only:
        total_batch_size_per_gpu = max(batch_size_mmug_cfg, batch_size_t2g_cfg)
        total_batch_size = (
            total_batch_size_per_gpu * accelerator.num_processes * config.training.gradient_accumulation_steps
        )
    else:
        total_batch_size_per_gpu = (config.training.batch_size_t2i
                                    + config.training.batch_size_lm
                                    + config.training.batch_size_mmu)
        total_batch_size = (
                (config.training.batch_size_t2i + config.training.batch_size_lm + config.training.batch_size_mmu)
                * accelerator.num_processes * config.training.gradient_accumulation_steps
        )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.pretrained_model_path, padding_side="left")
    use_vq_model = not mmug_only

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmug|>",  # multi-modal understanding for gene expression data
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob, use_reserved_token=True)

    print('special tokens : \n', uni_prompting.sptids_dict)

    # VQ model is only needed when visual generation/understanding flows are active.
    vq_model = None
    if use_vq_model:
        vq_model = get_vq_model_class(config.model.vq_model.type)
        if config.model.vq_model.get("pretrained_model_path", None):
            vq_model = vq_model().to(accelerator.device)
            state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
            vq_model.load_state_dict(state_dict)
        else:
            vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
        vq_model.eval()
        vq_model.requires_grad_(False)

    # Initialize mmada in pretraining stage 
    base_config = AutoConfig.from_pretrained(config.model.mmada.pretrained_model_path).to_dict()
    mmada_config_dict = {k: v for k, v in config.model.mmada.items()}
    merged_config = {**base_config, **mmada_config_dict}
    mmada_config = MMadaConfig(**merged_config)

    # Expand total vocabulary with a dedicated gene token range.
    # Gene ids should start after the text tokenizer space, not after the image codebook range.
    text_vocab_size = max(
        int(getattr(mmada_config, "llm_vocab_size", 0)),
        len(tokenizer),
    )
    base_vocab_size = max(
        text_vocab_size,
        max(reserved_token_mapping.values()) + 1,
    )
    gene_token_offset = int(base_vocab_size)
    gene_vocab_size = 0
    gene_vocab_num_embeddings = 0
    gene_vocab_max_id = None
    gene_vocab_path = config.dataset.params.get("gene_vocab_path", None)
    if gene_vocab_path is not None and os.path.exists(gene_vocab_path):
        _, gene_vocab_size, gene_vocab_num_embeddings, gene_vocab_max_id = load_gene_vocab(gene_vocab_path)
    total_vocab_size = gene_token_offset + gene_vocab_num_embeddings
    logger.info(
        f"Vocab expansion: text_vocab_size={text_vocab_size}, base_vocab={gene_token_offset}, "
        f"gene_vocab_size={gene_vocab_size}, gene_vocab_num_embeddings={gene_vocab_num_embeddings}, "
        f"total_vocab={total_vocab_size}, gene_token_offset={gene_token_offset}, "
        f"gene_vocab_max_id={gene_vocab_max_id}"
    )
    mmada_config.new_vocab_size = int(total_vocab_size)

    model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, torch_dtype=torch.bfloat16, config=mmada_config)
    model.resize_token_embeddings(mmada_config.new_vocab_size)
    model.config.embedding_size = model.config.vocab_size
    model = model.to(accelerator.device)

    # Initialize conditioning modules before optimizer creation so their parameters
    # are always included in optimizer param groups when enabled by the run config.
    model.init_gene_expression_value_encoder(
        hidden_dim=config.training.get("gene_expression_hidden_dim", None),
        dropout=float(config.training.get("gene_expression_dropout", 0.0)),
        max_value=float(config.training.get("gene_expression_max_value", 20.0)),
    )
    if config.dataset.params.get("cell_feature_root", None):
        model.init_cell_feature_soft_tokenizer(
            input_dim=int(config.dataset.params.get("cell_feature_dim", 768)),
            num_soft_tokens=int(config.training.get("cell_feature_num_soft_tokens", 4)),
            hidden_dim=config.training.get("cell_feature_hidden_dim", None),
            dropout=float(config.training.get("cell_feature_dropout", 0.0)),
        )

    mask_id = model.config.mask_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **args)
    else:
        mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size_t2i_without_accum = config.training.batch_size_t2i * accelerator.num_processes
    total_batch_size_t2i = (
            config.training.batch_size_t2i * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params
    train_dataloader_t2i = None
    train_dataloader_lm = None
    train_dataloader_mmu = None
    num_update_steps_per_epoch = None
    num_train_epochs = None

    if not mmug_only:
        # Data for generation
        if config.dataset.gen_type == "t2i":
            dataset = Text2ImageDataset(
                train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
                tokenizer=None,  # we want to get raw texts
                max_seq_length=preproc_config.max_seq_length,
                num_train_examples=config.experiment.max_train_examples_t2i,
                per_gpu_batch_size=config.training.batch_size_t2i,
                global_batch_size=total_batch_size_t2i_without_accum,
                num_workers=dataset_config.num_workers,
                resolution=preproc_config.resolution,
                shuffle_buffer_size=dataset_config.shuffle_buffer_size,
                pin_memory=dataset_config.pin_memory,
                persistent_workers=dataset_config.persistent_workers,
                external_caption_path=dataset_config.external_caption_path,
                external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
                external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
                external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
            )
            train_dataloader_t2i = dataset.train_dataloader
            num_update_steps_per_epoch = math.ceil(
                train_dataloader_t2i.num_batches / config.training.gradient_accumulation_steps)
            num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

        elif config.dataset.gen_type == "t2i_parquet":
            # this part relies on the internal packages, which will not be released
            num_update_steps_per_epoch = math.ceil(config.experiment.max_train_examples_t2i / total_batch_size_t2i)
            num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

            train_dataloader_t2i = create_imagetext_dataloader(
                train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
                batch_size=config.training.batch_size_t2i,
                image_size=preproc_config.resolution,
                num_workers=dataset_config.num_workers,
                num_readers=32,
                predefined_steps=num_update_steps_per_epoch,
                drop_last=True,
                shuffle=True,
                shuffle_buffer_size=dataset_config.shuffle_buffer_size
            )

        elif config.dataset.gen_type == "imagenet1k":
            dataset_imagenet = ImageNetDataset(
                dataset_config.train_t2i_shards_path_or_url,
                image_size=preproc_config.resolution,
            )

            print('process index : ',
                  accelerator.process_index, ', ', accelerator.num_processes,
                  "Length: ", len(dataset_imagenet))

            if accelerator.num_processes > 1:
                sampler = DistributedSampler(dataset_imagenet,
                                             num_replicas=accelerator.num_processes,
                                             rank=accelerator.process_index,
                                             shuffle=True,
                                             )
                shuffle = False
            else:
                sampler = None
                shuffle = True

            train_dataloader_t2i = DataLoader(dataset_imagenet, batch_size=config.training.batch_size_t2i,
                                              sampler=sampler, collate_fn=dataset_imagenet.collate_fn,
                                              shuffle=shuffle, num_workers=dataset_config.num_workers)
            num_update_steps_per_epoch = math.ceil(len(dataset_imagenet) / total_batch_size_t2i)
            num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

        else:
            raise ValueError(f"Unsupported dataset type {config.dataset.type}")

        total_batch_size_mmu_without_accum = config.training.batch_size_mmu * accelerator.num_processes
        # Data for image captioning
        if config.dataset.und_type == "captioning":
            dataset_mmu = Text2ImageDataset(
                train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
                tokenizer=None,  # we want to get raw texts
                max_seq_length=preproc_config.max_seq_length,
                num_train_examples=config.experiment.max_train_examples_mmu,
                per_gpu_batch_size=config.training.batch_size_mmu,
                global_batch_size=total_batch_size_mmu_without_accum,
                num_workers=dataset_config.num_workers,
                resolution=preproc_config.resolution,
                shuffle_buffer_size=dataset_config.shuffle_buffer_size,
                pin_memory=dataset_config.pin_memory,
                persistent_workers=dataset_config.persistent_workers,
                external_caption_path=dataset_config.external_caption_path,
                external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
                external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
                external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
                is_captioning=True,
                add_caption_prompt=dataset_config.add_caption_prompt,
            )
            train_dataloader_mmu = dataset_mmu.train_dataloader

        elif config.dataset.und_type == "captioning_parquet":
            train_dataloader_mmu = create_imagetext_dataloader(
                train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
                batch_size=config.training.batch_size_mmu,
                image_size=preproc_config.resolution,
                num_workers=dataset_config.num_workers,
                num_readers=32,
                predefined_steps=num_update_steps_per_epoch,
                drop_last=True,
                shuffle=True,
                shuffle_buffer_size=dataset_config.shuffle_buffer_size,
                is_captioning=True
            )

        else:
            raise NotImplementedError(f"Unsupported dataset type {config.dataset.und_type}")

    # CellwText gene-to-text dataset
    train_dataloader_t2g = None
    num_update_steps_per_epoch_t2g = None
    if hasattr(dataset_config, 'train_g2t_lmdb_path') and dataset_config.train_g2t_lmdb_path is not None:
        dataset_mmug = CellwTextDataset(
            lmdb_paths=dataset_config.train_g2t_lmdb_path,
            gene_vocab_path=dataset_config.get('gene_vocab_path', ''),
            celltype_label_path=dataset_config.get('celltype_label_path', None),
            tokenizer=tokenizer,
            max_seq_length=preproc_config.max_seq_length,
            max_gene_tokens=dataset_config.get('max_gene_tokens', 2000),
            num_expression_bins=dataset_config.get('num_expression_bins', 51),
            lmdb_vocab_path=dataset_config.get('lmdb_vocab_path', None),
            gene_token_offset=gene_token_offset,
            cell_metadata_path=dataset_config.get('cell_metadata_path', None),
            cell_feature_root=dataset_config.get('cell_feature_root', None),
            caption_template=dataset_config.get('caption_template', None),
            batch_size=batch_size_mmug_cfg,
            num_workers=dataset_config.num_workers,
            shuffle=False,
            pin_memory=dataset_config.pin_memory,
        )
        train_dataloader_mmug = dataset_mmug.get_dataloader()
        total_batch_size_mmug = (
            train_dataloader_mmug.batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
        )
        num_update_steps_per_epoch_mmug = math.ceil(len(dataset_mmug) / total_batch_size_mmug)

        if float(config.training.get("t2g_coeff", 0.0)) > 0.0:
            dataset_t2g = CellwTextDataset(
                lmdb_paths=dataset_config.train_g2t_lmdb_path,
                gene_vocab_path=dataset_config.get('gene_vocab_path', ''),
                celltype_label_path=dataset_config.get('celltype_label_path', None),
                tokenizer=tokenizer,
                max_seq_length=preproc_config.max_seq_length,
                max_gene_tokens=dataset_config.get('max_gene_tokens', 2000),
                num_expression_bins=dataset_config.get('num_expression_bins', 51),
                lmdb_vocab_path=dataset_config.get('lmdb_vocab_path', None),
                gene_token_offset=gene_token_offset,
                cell_metadata_path=dataset_config.get('cell_metadata_path', None),
                cell_feature_root=dataset_config.get('cell_feature_root', None),
                caption_template=dataset_config.get('caption_template', None),
                batch_size=batch_size_t2g_cfg,
                num_workers=dataset_config.num_workers,
                shuffle=False,
                pin_memory=dataset_config.pin_memory,
            )
            train_dataloader_t2g = dataset_t2g.get_dataloader()
            total_batch_size_t2g = (
                train_dataloader_t2g.batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
            )
            num_update_steps_per_epoch_t2g = math.ceil(len(dataset_t2g) / total_batch_size_t2g)
    else:
        train_dataloader_mmug = None
        num_update_steps_per_epoch_mmug = None

    if not mmug_only:
        # LLM pure text dataset: RefinedWeb
        dataset_lm = RefinedWebDataset(data_path=dataset_config.train_lm_shards_path_or_url,
                                       rank=accelerator.process_index,
                                       world_size=accelerator.num_processes,
                                       num_workers=dataset_config.num_workers)

        train_dataloader_lm = torch.utils.data.DataLoader(dataset_lm, batch_size=config.training.batch_size_lm,
                                                          sampler=None, collate_fn=dataset_lm.collate_fn,
                                                          num_workers=dataset_config.num_workers)

    # Combine these dataloaders into a single iterable model
    if mmug_only:
        if train_dataloader_mmug is None:
            raise ValueError("mmug_only=True but no mmug dataloader is available. Check train_g2t_lmdb_path.")
        if num_update_steps_per_epoch_mmug is None:
            raise ValueError("mmug_only=True but num_update_steps_per_epoch_mmug is None.")
        iterables = {"mmug_flow": train_dataloader_mmug}
        step_candidates = [num_update_steps_per_epoch_mmug]
        if train_dataloader_t2g is not None:
            iterables["t2g_flow"] = train_dataloader_t2g
            step_candidates.append(num_update_steps_per_epoch_t2g)
        if config.dataset.combined_loader_mode == "max_size_cycle":
            num_update_steps_per_epoch = max(step_candidates)
        elif config.dataset.combined_loader_mode == "min_size":
            num_update_steps_per_epoch = min(step_candidates)
        else:
            num_update_steps_per_epoch = step_candidates[0]
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
    else:
        iterables = {
            "t2i_flow": train_dataloader_t2i,
            "lm_flow": train_dataloader_lm,
            "mmu_flow": train_dataloader_mmu,
        }
        if train_dataloader_mmug is not None:
            iterables["mmug_flow"] = train_dataloader_mmug  # multi-modal understanding for gene data
        if train_dataloader_t2g is not None:
            iterables["t2g_flow"] = train_dataloader_t2g

    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0
    resume_step_in_epoch = 0
    resume_path = None
    resume_with_full_state = False
    # Resume old checkpoints with strict=False by default, because new conditioning modules
    # (e.g., gene_expression_value_encoder / cell feature tokenizer) may be absent in older runs.
    checkpoint_strict = bool(config.experiment.get("checkpoint_strict", False))
    logger.info(f"Checkpoint strict loading: {checkpoint_strict}")

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        logger.info(f"dirs: {dirs}")
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        logger.info(f"path: {path}")
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)
            resume_path = path
            logger.info(f"Resuming from checkpoint: {path}")
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step_in_epoch = global_step % num_update_steps_per_epoch

            full_state_dir = os.path.join(path, "training_state")
            current_vocab_size = model.get_input_embeddings().weight.shape[0]
            checkpoint_vocab_size = None
            restore_vocab_size_after_load = None
            checkpoint_cfg_path = os.path.join(path, "unwrapped_model", "config.json")
            if os.path.exists(checkpoint_cfg_path):
                try:
                    with open(checkpoint_cfg_path, "r") as f:
                        checkpoint_vocab_size = int(json.load(f).get("vocab_size", current_vocab_size))
                except Exception:
                    checkpoint_vocab_size = None

            if checkpoint_vocab_size is not None and checkpoint_vocab_size != current_vocab_size:
                logger.warning(
                    f"Checkpoint vocab ({checkpoint_vocab_size}) != current vocab ({current_vocab_size}). "
                    f"Will load at checkpoint size, then re-expand to current size."
                )
                restore_vocab_size_after_load = current_vocab_size
                model.resize_token_embeddings(checkpoint_vocab_size)
                model.config.embedding_size = model.config.vocab_size

            if os.path.isdir(full_state_dir) and (checkpoint_vocab_size is None or checkpoint_vocab_size == current_vocab_size):
                resume_with_full_state = True
                logger.info(f"Found full training state at {full_state_dir}; will restore after accelerator.prepare")
            elif os.path.exists(f'{path}/unwrapped_model/pytorch_model.bin'):
                state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
                load_result = model.load_state_dict(state_dict, strict=checkpoint_strict)
                if not checkpoint_strict:
                    logger.warning(
                        "Loaded checkpoint with strict=False to allow newly added conditioning modules. "
                        f"Missing keys: {list(load_result.missing_keys)[:20]}"
                    )
                del state_dict
                if restore_vocab_size_after_load is not None:
                    model.resize_token_embeddings(restore_vocab_size_after_load)
                    model.config.embedding_size = model.config.vocab_size
            elif os.path.exists(f'{path}/unwrapped_model/pytorch_model.bin.index.json'):
                from safetensors.torch import load_file
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(model, f'{path}/unwrapped_model/', strict=checkpoint_strict)
                if not checkpoint_strict:
                    logger.warning(
                        "Loaded sharded checkpoint with strict=False to allow newly added conditioning modules."
                    )
                if restore_vocab_size_after_load is not None:
                    model.resize_token_embeddings(restore_vocab_size_after_load)
                    model.config.embedding_size = model.config.vocab_size
            # if safetensors sharded checkpoint exists
            elif os.path.exists(f'{path}/unwrapped_model/model.safetensors.index.json'):
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(
                    model, 
                    f'{path}/unwrapped_model/',
                    strict=checkpoint_strict,
                    # weight_map=None, 
                    # load_state_dict_fn="safetensors"
                )
                if not checkpoint_strict:
                    logger.warning(
                        "Loaded safetensors sharded checkpoint with strict=False to allow newly added conditioning modules."
                    )
                if restore_vocab_size_after_load is not None:
                    model.resize_token_embeddings(restore_vocab_size_after_load)
                    model.config.embedding_size = model.config.vocab_size
            else:
                raise FileNotFoundError(f"Checkpoint {path}/unwrapped_model/pytorch_model.bin not found")
    else:
        logger.info("Not resuming from checkpoint")

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    if resume_with_full_state and resume_path is not None:
        state_dir = os.path.join(resume_path, "training_state")
        logger.info(f"Loading full training state from {state_dir}")
        accelerator.load_state(state_dir)

    if vq_model is not None:
        vq_model.to(device=accelerator.device)

    mask_dtype = model.get_input_embeddings().weight.dtype

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    @torch.no_grad()
    def prepare_inputs_and_labels(
            pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
            texts: Union[str, str],
            min_masking_rate: float = 0.0,
            is_train: bool = True,
    ):

        image_tokens = vq_model.get_code(pixel_values_or_image_ids)
        image_tokens = image_tokens + len(uni_prompting.text_tokenizer)
        # create MLM mask and labels
        input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
            image_tokens,
            mask_id,
            config,
            mask_schedule=mask_schedule,
            is_train=is_train,
        )
        input_ids, masks, labels = uni_prompting((texts, input_ids, labels), 't2i')
        return input_ids, labels, mask_prob, image_tokens, masks
    
    @torch.no_grad()
    def prepare_inputs_and_labels_for_text(
        texts: Union[str, str], max_seq_len, eps=1e-3
    ):
        # create MLM mask and labels
        
        input_ids_lm, prompt_mask, labels_lm = uni_prompting((texts_lm, max_seq_len), 'lm')
        b, l = input_ids_lm.shape
        t = torch.rand(b, device=input_ids_lm.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_lm.device) < p_mask
        # 126336 is used for [MASK] token
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_lm)
        masked_indices = noisy_batch == mask_id 
        
        return noisy_batch, labels_lm, p_mask
    
    @torch.no_grad()
    def prepare_inputs_and_labels_for_mmu(
        input_ids_mmu, prompt_masks, labels_mmu, eps=1e-3
    ):
        b, l = input_ids_mmu.shape
        t = torch.rand(b, device=input_ids_mmu.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_mmu.device) < p_mask
        # 126336 is used for [MASK] token 
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_mmu)
        masked_indices = noisy_batch == mask_id 
        noisy_batch[prompt_masks.bool()] = input_ids_mmu[prompt_masks.bool()]
        masked_indices = noisy_batch == mask_id 

        prompt_masks = prompt_masks.to(torch.int64)    
        answer_lengths = torch.sum((1 - prompt_masks), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])    

        return noisy_batch, labels_mmu, p_mask, answer_lengths

    @torch.no_grad()
    def prepare_inputs_and_labels_for_t2g(
        gene_ids, texts, eps=1e-3
    ):
        input_ids_t2g, prompt_masks_t2g, labels_t2g = uni_prompting((gene_ids, texts), 't2g')
        b, l = input_ids_t2g.shape
        gene_len = gene_ids.shape[1]

        # t2g prompt layout is fixed: [task] [text condition] [soi] [gene tokens] [eoi]
        # so the target segment is always the final gene_len tokens before the closing <|eoi|>.
        target_masks_t2g = torch.zeros((b, l), dtype=torch.bool, device=input_ids_t2g.device)
        target_start = l - gene_len - 1
        target_end = l - 1
        if target_start < 0:
            raise ValueError(f"Invalid t2g layout: seq_len={l}, gene_len={gene_len}")
        target_masks_t2g[:, target_start:target_end] = True

        labels_t2g = labels_t2g.clone()
        labels_t2g[~target_masks_t2g] = -100

        t = torch.rand(b, device=input_ids_t2g.device)
        p_mask_t2g = (1 - eps) * t + eps
        p_mask_t2g = p_mask_t2g[:, None].repeat(1, l)

        sampled_masks_t2g = (torch.rand((b, l), device=input_ids_t2g.device) < p_mask_t2g) & target_masks_t2g

        # Ensure every sample contributes at least one masked gene token.
        fallback_pos = target_start
        for i in range(b):
            if sampled_masks_t2g[i].any():
                continue
            sampled_masks_t2g[i, fallback_pos] = True

        noisy_batch_t2g = input_ids_t2g.clone()
        noisy_batch_t2g[sampled_masks_t2g] = mask_id

        answer_lengths_t2g = target_masks_t2g.to(torch.int64).sum(dim=-1, keepdim=True)
        answer_lengths_t2g = answer_lengths_t2g.repeat(1, l)

        return noisy_batch_t2g, labels_t2g, p_mask_t2g, answer_lengths_t2g

    def compute_masked_diffusion_loss(logits, input_ids_masked, labels_masked, p_mask_masked, answer_lengths_masked):
        masked_indices = input_ids_masked == mask_id
        if masked_indices.any():
            p_mask_for_loss = p_mask_masked.to(masked_indices.device)
            answer_lengths_for_loss = answer_lengths_masked.to(masked_indices.device)
            loss_masked = torch.nn.functional.cross_entropy(
                logits[masked_indices].contiguous().view(-1, logits.shape[-1]),
                labels_masked[masked_indices].contiguous().view(-1),
                ignore_index=-100,
                reduction='none',
            ) / p_mask_for_loss[masked_indices]
            loss_masked = torch.sum(
                loss_masked / answer_lengths_for_loss[masked_indices]
            ) / logits.shape[0]
        else:
            loss_masked = logits.sum() * 0.0
        return loss_masked



    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        epoch_iterator = combined_dataloader
        if epoch == first_epoch and resume_step_in_epoch > 0:
            skip_batches = resume_step_in_epoch * config.training.gradient_accumulation_steps
            logger.info(
                f"Resuming dataloader position: skipping {skip_batches} micro-batches "
                f"({resume_step_in_epoch} optimizer steps) in epoch {epoch}"
            )
            epoch_iterator = itertools.islice(combined_dataloader, skip_batches, None)
            resume_step_in_epoch = 0

        for batch, batch_idx, dataloader_idx in epoch_iterator:
            # for loss calculation
            batch_size_t2i = batch["t2i_flow"]["images"].shape[0] if "t2i_flow" in batch else 0
            batch_size_mmug = batch["mmug_flow"]["gene_ids"].shape[0] if "mmug_flow" in batch else 0
            batch_size_t2g = batch["t2g_flow"]["gene_ids"].shape[0] if "t2g_flow" in batch else 0
            batch_size_lm = len(batch["lm_flow"]["input_ids"]) if "lm_flow" in batch else 0
            batch_size_mmu = batch["mmu_flow"]["images"].shape[0] if "mmu_flow" in batch else 0

            if mmug_only:
                data_time_m.update(time.time() - end)
                if batch_size_mmug == 0:
                    continue

                gene_ids = batch["mmug_flow"]["gene_ids"].to(accelerator.device, non_blocking=True)
                gene_expression_mmug = batch["mmug_flow"].get("gene_expression", None)
                if gene_expression_mmug is not None:
                    gene_expression_mmug = gene_expression_mmug.to(accelerator.device, non_blocking=True)
                texts_mmug = batch["mmug_flow"].get("texts", [""] * gene_ids.shape[0])
                cell_features_mmug = batch["mmug_flow"].get("cell_features", None)
                if cell_features_mmug is not None:
                    cell_features_mmug = cell_features_mmug.to(accelerator.device, non_blocking=True)
                input_ids_mmug, prompt_masks_mmug, labels_mmug = uni_prompting((texts_mmug, gene_ids), 'mmug')
                (
                    input_ids_mmug,
                    labels_mmug,
                    p_mask_mmug,
                    answer_lengths_mmug,
                ) = prepare_inputs_and_labels_for_mmu(input_ids_mmug, prompt_masks_mmug, labels_mmug)
                input_ids_mmug = input_ids_mmug.to(accelerator.device, non_blocking=True)
                labels_mmug = labels_mmug.to(accelerator.device, non_blocking=True)
                text_to_gene_coeff = float(config.training.get("t2g_coeff", 0.0))
                input_ids_t2g = None
                labels_t2g = None
                p_mask_t2g = None
                answer_lengths_t2g = None
                if text_to_gene_coeff > 0.0 and "t2g_flow" in batch:
                    gene_ids_t2g = batch["t2g_flow"]["gene_ids"].to(accelerator.device, non_blocking=True)
                    texts_t2g = batch["t2g_flow"].get("texts", [""] * gene_ids_t2g.shape[0])
                    (
                        input_ids_t2g,
                        labels_t2g,
                        p_mask_t2g,
                        answer_lengths_t2g,
                    ) = prepare_inputs_and_labels_for_t2g(gene_ids_t2g, texts_t2g)
                    input_ids_t2g = input_ids_t2g.to(accelerator.device, non_blocking=True)
                    labels_t2g = labels_t2g.to(accelerator.device, non_blocking=True)

                if global_step == 0 and epoch == 0:
                    logger.info("MMUG input ids: {}".format(input_ids_mmug))
                    logger.info("MMUG labels: {}".format(labels_mmug))
                    if input_ids_t2g is not None:
                        logger.info("T2G input ids: {}".format(input_ids_t2g))
                        logger.info("T2G labels: {}".format(labels_t2g))

                with accelerator.accumulate(model):
                    input_ids_for_loss = input_ids_mmug
                    labels_for_loss = labels_mmug
                    p_mask_for_loss = p_mask_mmug
                    answer_lengths_for_loss = answer_lengths_mmug
                    attention_bias_mmug = torch.ones(
                        input_ids_mmug.shape[0], 1, input_ids_mmug.shape[1], input_ids_mmug.shape[1], device=input_ids_mmug.device
                    )

                    model_kwargs = {"attention_bias": attention_bias_mmug}
                    if cell_features_mmug is not None:
                        unwrapped_model = accelerator.unwrap_model(model)
                        cell_feature_soft_tokenizer = getattr(unwrapped_model, "cell_feature_soft_tokenizer", None)
                        if cell_feature_soft_tokenizer is None:
                            raise RuntimeError(
                                "cell_feature_soft_tokenizer must be initialized before optimizer creation when cell features are enabled."
                            )
                        expected_num_soft_tokens = int(config.training.get("cell_feature_num_soft_tokens", 4))
                        if cell_feature_soft_tokenizer.num_soft_tokens != expected_num_soft_tokens:
                            raise ValueError(
                                f"Configured cell_feature_num_soft_tokens={expected_num_soft_tokens}, "
                                f"but initialized projector uses {cell_feature_soft_tokenizer.num_soft_tokens}."
                            )
                        if cell_feature_soft_tokenizer.input_dim != cell_features_mmug.shape[-1]:
                            raise ValueError(
                                f"Cell feature dim mismatch: projector expects {cell_feature_soft_tokenizer.input_dim}, "
                                f"but batch provides {cell_features_mmug.shape[-1]}."
                            )
                        if getattr(unwrapped_model, "gene_expression_value_encoder", None) is None:
                            raise RuntimeError(
                                "gene_expression_value_encoder must be initialized before optimizer creation."
                            )

                        prefix_length = cell_feature_soft_tokenizer.num_soft_tokens
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
                        answer_lengths_for_loss = torch.cat([torch.ones_like(prefix_ids, dtype=answer_lengths_mmug.dtype), answer_lengths_mmug], dim=1)
                        attention_bias_mmug = unwrapped_model.extend_attention_bias_for_prefix(attention_bias_mmug, prefix_length)
                        inputs_embeds_mmug = unwrapped_model.build_inputs_embeds_with_conditioning(
                            input_ids=input_ids_mmug,
                            cell_features=cell_features_mmug,
                            gene_expression=gene_expression_mmug,
                            gene_token_start=2,
                        )
                        model_kwargs = {
                            "input_ids": input_ids_for_loss,
                            "inputs_embeds": inputs_embeds_mmug,
                            "attention_bias": attention_bias_mmug,
                        }
                    else:
                        if gene_expression_mmug is not None:
                            unwrapped_model = accelerator.unwrap_model(model)
                            if getattr(unwrapped_model, "gene_expression_value_encoder", None) is None:
                                raise RuntimeError(
                                    "gene_expression_value_encoder must be initialized before optimizer creation."
                                )
                            inputs_embeds_mmug = unwrapped_model.build_inputs_embeds_with_conditioning(
                                input_ids=input_ids_mmug,
                                gene_expression=gene_expression_mmug,
                                gene_token_start=2,
                            )
                            model_kwargs = {
                                "input_ids": input_ids_mmug,
                                "inputs_embeds": inputs_embeds_mmug,
                                "attention_bias": attention_bias_mmug,
                            }
                        else:
                            model_kwargs = {
                                "input_ids": input_ids_mmug,
                                "attention_bias": attention_bias_mmug,
                            }

                    logits = model(**model_kwargs).logits
                    loss_mmug = compute_masked_diffusion_loss(
                        logits,
                        input_ids_for_loss,
                        labels_for_loss,
                        p_mask_for_loss,
                        answer_lengths_for_loss,
                    )

                    if input_ids_t2g is not None:
                        attention_bias_t2g = torch.ones(
                            input_ids_t2g.shape[0], 1, input_ids_t2g.shape[1], input_ids_t2g.shape[1], device=input_ids_t2g.device
                        )
                        logits_t2g = model(input_ids=input_ids_t2g, attention_bias=attention_bias_t2g).logits
                        masked_t2g = input_ids_t2g == mask_id
                        valid_t2g = masked_t2g & (labels_t2g != -100)
                        t2g_masked_tokens = masked_t2g.sum().detach()
                        t2g_valid_masked_tokens = valid_t2g.sum().detach()
                        loss_t2g = compute_masked_diffusion_loss(
                            logits_t2g,
                            input_ids_t2g,
                            labels_t2g,
                            p_mask_t2g,
                            answer_lengths_t2g,
                        )
                    else:
                        loss_t2g = loss_mmug.new_zeros(())
                        t2g_masked_tokens = torch.zeros((), device=loss_mmug.device, dtype=torch.long)
                        t2g_valid_masked_tokens = torch.zeros((), device=loss_mmug.device, dtype=torch.long)

                    avg_loss_mmug = accelerator.gather(loss_mmug.repeat(batch_size_mmug_cfg)).mean()
                    avg_loss_t2g = accelerator.gather(loss_t2g.repeat(max(batch_size_t2g_cfg, 1))).mean()
                    avg_t2g_masked_tokens = accelerator.gather(t2g_masked_tokens.reshape(1)).float().mean()
                    avg_t2g_valid_masked_tokens = accelerator.gather(t2g_valid_masked_tokens.reshape(1)).float().mean()
                    mmug_coeff = (config.training.mmug_coeff if hasattr(config.training, "mmug_coeff") else config.training.g2t_coeff)
                    loss_gene = mmug_coeff * loss_mmug + text_to_gene_coeff * loss_t2g
                    loss = loss_gene.mean()
                    avg_loss_gene = accelerator.gather(loss_gene.detach().reshape(1)).mean()
                    avg_loss_total = accelerator.gather(loss.detach().reshape(1)).mean()

                    accelerator.backward(loss)

                    if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()

                    if (
                            accelerator.sync_gradients
                            and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                            and accelerator.is_main_process
                    ):
                        log_grad_norm(model, accelerator, global_step + 1)

                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    batch_time_m.update(time.time() - end)
                    end = time.time()

                    if (global_step + 1) % config.experiment.log_every == 0:
                        samples_per_second_per_gpu = (
                                config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                        )
                        logs = {
                            "step_loss_mmug": avg_loss_mmug.item(),
                            "step_loss_t2g": avg_loss_t2g.item(),
                            "step_loss_gene": avg_loss_gene.item(),
                            "step_loss_total": avg_loss_total.item(),
                            "t2g_masked_tokens": avg_t2g_masked_tokens.item(),
                            "t2g_valid_masked_tokens": avg_t2g_valid_masked_tokens.item(),
                            "mmug_batch_size": int(batch_size_mmug),
                            "t2g_batch_size": int(batch_size_t2g),
                            "mmug_active": int(batch_size_mmug > 0),
                            "lr": lr_scheduler.get_last_lr()[0],
                            "samples/sec/gpu": samples_per_second_per_gpu,
                            "data_time": data_time_m.val,
                            "batch_time": batch_time_m.val,
                        }
                        accelerator.log(logs, step=global_step + 1)

                        logger.info(
                            f"Step: {global_step + 1} "
                            f"Loss_mmug: {avg_loss_mmug.item():0.4f} "
                            f"Loss_t2g: {avg_loss_t2g.item():0.4f} "
                            f"Loss_gene: {avg_loss_gene.item():0.4f} "
                            f"Loss_total: {avg_loss_total.item():0.4f} "
                            f"T2G_Masked: {avg_t2g_masked_tokens.item():0.1f} "
                            f"T2G_Valid: {avg_t2g_valid_masked_tokens.item():0.1f} "
                            f"MMUG_BS: {batch_size_mmug} "
                            f"T2G_BS: {batch_size_t2g} "
                            f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_m.val:0.4f} "
                            f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                        )

                        batch_time_m.reset()
                        data_time_m.reset()

                    if (global_step + 1) % config.experiment.save_every == 0:
                        save_checkpoint(model, config, accelerator, global_step + 1)

                    global_step += 1

                if global_step >= config.training.max_train_steps:
                    break

                continue

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for class-conditional/text-to-image generation
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            pixel_values, texts = batch["t2i_flow"]["images"], batch["t2i_flow"]["input_ids"]
            pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
            data_time_m.update(time.time() - end)

            # Encode images to image tokens, mask them and create input and labels
            (
                input_ids,
                labels,
                mask_prob,
                image_tokens_ori,
                t2i_masks
            ) = prepare_inputs_and_labels(pixel_values, texts, config.training.min_masking_rate)

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for language modeling
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            max_seq_len = input_ids.shape[-1]
            texts_lm = batch["lm_flow"]["input_ids"]
            (
                input_ids_lm,  
                labels_lm,
                p_mask_lm
            ) = prepare_inputs_and_labels_for_text(texts_lm, max_seq_len)  
            input_ids = torch.cat((input_ids, input_ids_lm.to(input_ids.device)), dim=0)
            labels = torch.cat((labels, labels_lm.to(input_ids.device)), dim=0)

            


            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for gene-to-text
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            input_ids_mmug = None
            labels_mmug = None
            p_mask_mmug = None
            answer_lengths_mmug = None
            input_ids_t2g = None
            labels_t2g = None
            p_mask_t2g = None
            answer_lengths_t2g = None
            if "mmug_flow" in batch:
                gene_ids = batch["mmug_flow"]["gene_ids"].to(accelerator.device, non_blocking=True)
                texts_mmug = batch["mmug_flow"].get("texts", [""] * gene_ids.shape[0])

                # Use a dedicated forward for mmug so we can keep long gene sequences (e.g. 2000 tokens).
                input_ids_mmug, prompt_masks_mmug, labels_mmug = uni_prompting((texts_mmug, gene_ids), 'mmug')
                (
                    input_ids_mmug,
                    labels_mmug,
                    p_mask_mmug,
                    answer_lengths_mmug,
                ) = prepare_inputs_and_labels_for_mmu(input_ids_mmug, prompt_masks_mmug, labels_mmug)
                input_ids_mmug = input_ids_mmug.to(accelerator.device, non_blocking=True)
                labels_mmug = labels_mmug.to(accelerator.device, non_blocking=True)

            if float(config.training.get("t2g_coeff", 0.0)) > 0.0 and "t2g_flow" in batch:
                gene_ids_t2g = batch["t2g_flow"]["gene_ids"].to(accelerator.device, non_blocking=True)
                texts_t2g = batch["t2g_flow"].get("texts", [""] * gene_ids_t2g.shape[0])
                (
                    input_ids_t2g,
                    labels_t2g,
                    p_mask_t2g,
                    answer_lengths_t2g,
                ) = prepare_inputs_and_labels_for_t2g(gene_ids_t2g, texts_t2g)
                input_ids_t2g = input_ids_t2g.to(accelerator.device, non_blocking=True)
                labels_t2g = labels_t2g.to(accelerator.device, non_blocking=True)

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for captioning/multimodal understanding
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            if "llava" in config.dataset.und_type:
                pixel_values_mmu, input_ids_mmu, labels_mmu = (batch["mmu_flow"]["images"], batch["mmu_flow"]["input_ids"],batch["mmu_flow"]["labels"])
                pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
                input_ids_mmu = input_ids_mmu.to(accelerator.device, non_blocking=True)
                image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)

                input_ids_mmu = torch.cat([
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(
                        accelerator.device),
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(
                        accelerator.device),
                    image_tokens_mmu,
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(
                        accelerator.device),
                    input_ids_mmu,
                ], dim=1).long()

                labels_mmu = torch.cat([
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                    torch.ones_like(image_tokens_mmu) * uni_prompting.ignore_id,
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                    labels_mmu.to(accelerator.device)
                ], dim=1).long()

            else:
                pixel_values_mmu, texts_mmu = batch["mmu_flow"]["images"], batch["mmu_flow"]["input_ids"]
                pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
                image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)
                
                input_ids_mmu, prompt_masks, labels_mmu = uni_prompting((image_tokens_mmu, texts_mmu), 'mmu')
                (
                    input_ids_mmu,  
                    labels_mmu,
                    p_mask_mmu,
                    answer_lengths
                ) = prepare_inputs_and_labels_for_mmu(input_ids_mmu, prompt_masks, labels_mmu)
                input_ids_mmu = input_ids_mmu.to(accelerator.device, non_blocking=True)

            # Pad all flows to a unified sequence length before concatenation.
            target_seq_len = max(
                input_ids.shape[1],
                input_ids_mmu.shape[1],
                input_ids_mmug.shape[1] if input_ids_mmug is not None else 0,
            )
            target_seq_len = max(target_seq_len, int(config.training.get("unified_seq_len", 0)))

            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

            def _pad_to_len(x, pad_value):
                if x is None:
                    return None
                cur_len = x.shape[1]
                if cur_len >= target_seq_len:
                    return x
                pad = x.new_full((x.shape[0], target_seq_len - cur_len), pad_value)
                return torch.cat((x, pad), dim=1)

            input_ids = _pad_to_len(input_ids, pad_token_id)
            labels = _pad_to_len(labels, -100)
            t2i_masks = _pad_to_len(t2i_masks, 0)
            input_ids_mmu = _pad_to_len(input_ids_mmu.to(input_ids.device), pad_token_id)
            labels_mmu = _pad_to_len(labels_mmu.to(input_ids.device), -100)
            p_mask_lm = _pad_to_len(p_mask_lm, 0.0)
            p_mask_mmu = _pad_to_len(p_mask_mmu, 0.0)
            answer_lengths = _pad_to_len(answer_lengths, 1)

            if input_ids_mmug is not None:
                input_ids_mmug = _pad_to_len(input_ids_mmug.to(input_ids.device), pad_token_id)
                labels_mmug = _pad_to_len(labels_mmug.to(input_ids.device), -100)
                p_mask_mmug = _pad_to_len(p_mask_mmug, 0.0)
                answer_lengths_mmug = _pad_to_len(answer_lengths_mmug, 1)
                input_ids = torch.cat((input_ids, input_ids_mmug), dim=0)
                labels = torch.cat((labels, labels_mmug), dim=0)
            else:
                p_mask_mmug = None
                answer_lengths_mmug = None

            input_ids = torch.cat((input_ids, input_ids_mmu), dim=0)
            labels = torch.cat((labels, labels_mmu), dim=0)

            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(labels))

            with accelerator.accumulate(model):
                logits, loss_t2i, loss_mmug, loss_lm, loss_mmu = model.forward_process(
                    input_ids=input_ids,
                    labels=labels,
                    batch_size_t2i=batch_size_t2i,
                    batch_size_mmug=batch_size_mmug,
                    batch_size_lm=batch_size_lm,
                    batch_size_mmu=batch_size_mmu,
                    max_seq_length=config.dataset.preprocessing.max_seq_length,
                    p_mask_lm=p_mask_lm,
                    p_mask_mmu=p_mask_mmu,
                    p_mask_mmug=p_mask_mmug,
                    answer_lengths=answer_lengths,
                    answer_lengths_mmug=answer_lengths_mmug,
                    t2i_masks=t2i_masks
                )
                if input_ids_t2g is not None:
                    attention_bias_t2g = torch.ones(
                        input_ids_t2g.shape[0], 1, input_ids_t2g.shape[1], input_ids_t2g.shape[1], device=input_ids_t2g.device
                    )
                    logits_t2g = model(input_ids=input_ids_t2g, attention_bias=attention_bias_t2g).logits
                    masked_t2g = input_ids_t2g == mask_id
                    valid_t2g = masked_t2g & (labels_t2g != -100)
                    t2g_masked_tokens = masked_t2g.sum().detach()
                    t2g_valid_masked_tokens = valid_t2g.sum().detach()
                    loss_t2g = compute_masked_diffusion_loss(
                        logits_t2g,
                        input_ids_t2g,
                        labels_t2g,
                        p_mask_t2g,
                        answer_lengths_t2g,
                    )
                else:
                    loss_t2g = loss_t2i.new_zeros(())
                    t2g_masked_tokens = torch.zeros((), device=loss_t2i.device, dtype=torch.long)
                    t2g_valid_masked_tokens = torch.zeros((), device=loss_t2i.device, dtype=torch.long)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss_t2i = accelerator.gather(loss_t2i.repeat(config.training.batch_size_t2i)).mean()
                if batch_size_mmug > 0:
                    mmug_bs_for_log = (
                        config.training.batch_size_mmug if hasattr(config.training, "batch_size_mmug")
                        else (
                            config.training.batch_size_g2t if hasattr(config.training, "batch_size_g2t")
                            else config.training.batch_size_t2i
                        )
                    )
                    avg_loss_mmug = accelerator.gather(loss_mmug.repeat(mmug_bs_for_log)).mean()
                    avg_loss_t2g = accelerator.gather(loss_t2g.repeat(max(batch_size_t2g_cfg, 1))).mean()
                else:
                    avg_loss_mmug = torch.tensor(0.0, device=loss_t2i.device)
                    avg_loss_t2g = torch.tensor(0.0, device=loss_t2i.device)
                avg_t2g_masked_tokens = accelerator.gather(t2g_masked_tokens.reshape(1)).float().mean()
                avg_t2g_valid_masked_tokens = accelerator.gather(t2g_valid_masked_tokens.reshape(1)).float().mean()
                avg_loss_lm = accelerator.gather(loss_lm.repeat(config.training.batch_size_lm)).mean()
                avg_loss_mmu = accelerator.gather(loss_mmu.repeat(config.training.batch_size_mmu)).mean()
                mmug_coeff = (config.training.mmug_coeff if hasattr(config.training, "mmug_coeff") else config.training.g2t_coeff)
                t2g_coeff = float(config.training.get("t2g_coeff", 0.0))
                loss_gene = mmug_coeff * loss_mmug + t2g_coeff * loss_t2g
                loss = config.training.t2i_coeff * loss_t2i + \
                       loss_gene + \
                       config.training.lm_coeff * loss_lm + \
                       config.training.mmu_coeff * loss_mmu

                avg_loss_gene = accelerator.gather(loss_gene.detach().reshape(1)).mean()
                avg_loss_total = accelerator.gather(loss.detach().reshape(1)).mean()
                avg_masking_rate = accelerator.gather(mask_prob.repeat(config.training.batch_size_t2i)).mean()

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        "step_loss_t2i": avg_loss_t2i.item(),
                        "step_loss_mmug": avg_loss_mmug.item(),
                        "step_loss_t2g": avg_loss_t2g.item(),
                        "step_loss_gene": avg_loss_gene.item(),
                        "step_loss_total": avg_loss_total.item(),
                        "t2g_masked_tokens": avg_t2g_masked_tokens.item(),
                        "t2g_valid_masked_tokens": avg_t2g_valid_masked_tokens.item(),
                        "mmug_batch_size": int(batch_size_mmug),
                        "t2g_batch_size": int(batch_size_t2g),
                        "mmug_active": int(batch_size_mmug > 0),
                        "step_loss_mmu": avg_loss_mmu.item(),
                        "step_loss_lm": avg_loss_lm.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_t2i: {avg_loss_t2i.item():0.4f} "
                        f"Loss_mmug: {avg_loss_mmug.item():0.4f} "
                        f"Loss_t2g: {avg_loss_t2g.item():0.4f} "
                        f"Loss_gene: {avg_loss_gene.item():0.4f} "
                        f"Loss_total: {avg_loss_total.item():0.4f} "
                        f"T2G_Masked: {avg_t2g_masked_tokens.item():0.1f} "
                        f"T2G_Valid: {avg_t2g_valid_masked_tokens.item():0.1f} "
                        f"MMUG_BS: {batch_size_mmug} "
                        f"T2G_BS: {batch_size_t2g} "
                        f"Loss_mmu: {avg_loss_mmu.item():0.4f} "
                        f"Loss_lm: {avg_loss_lm.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                if ((global_step + 1) % config.experiment.generate_every == 0 or global_step == 0) and accelerator.is_main_process:
                    generate_images(
                        model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                        mask_schedule=mask_schedule,
                    )

                    visualize_predictions(
                        model,
                        vq_model,
                        uni_prompting,
                        config,
                        global_step + 1,
                        input_ids,
                        image_tokens_ori,
                        batch["t2i_flow"]["images"],
                        texts,
                        logits,
                        accelerator
                    )
                    
                    understanding_images(
                        model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                    )

                global_step += 1

            if global_step >= config.training.max_train_steps:
                break

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, config, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=True)

    accelerator.end_training()


@torch.no_grad()
def visualize_predictions(
        model,
        vq_model,
        uni_prompting,
        config,
        global_step,
        input_ids,
        image_tokens_ori,
        ori_images,
        texts,
        logits,
        accelerator
):
    logger.info("Visualizing predictions...")
    model.eval()

    recons_images = vq_model.decode_code(image_tokens_ori - len(uni_prompting.text_tokenizer))
    recons_images = torch.clamp((recons_images + 1.0) / 2.0, min=0.0, max=1.0)
    recons_images *= 255.0
    recons_images = recons_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    images = torch.clamp((ori_images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    predictions = logits[:config.training.batch_size_t2i, -(config.model.mmada.num_vq_tokens + 1):-1:, len(uni_prompting.text_tokenizer) + config.model.mmada.num_new_special_tokens: len(uni_prompting.text_tokenizer) + config.model.mmada.num_new_special_tokens + config.model.mmada.codebook_size]
    
    predictions = predictions.argmax(axis=-1)
    mask_token_id = accelerator.unwrap_model(model).config.mask_token_id - len(uni_prompting.text_tokenizer)
    input_ids = input_ids[:config.training.batch_size_t2i, -(config.model.mmada.num_vq_tokens + 1):-1:] - len(uni_prompting.text_tokenizer)
    mask_ratio = list((torch.where(input_ids == mask_token_id, 1, 0).sum(
        dim=-1) / config.model.mmada.num_vq_tokens).cpu().numpy())
    predicted_images = torch.where(input_ids == mask_token_id, predictions, input_ids)
    predicted_images = vq_model.decode_code(predicted_images)
    predicted_images = torch.clamp((predicted_images + 1.0) / 2.0, min=0.0, max=1.0)
    predicted_images *= 255.0
    predicted_images = predicted_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    predicted_images = np.concatenate((images, recons_images, predicted_images), 2)
    pil_images = [Image.fromarray(image) for image in predicted_images]

    # Log images
    try:
        wandb_images = [wandb.Image(image, caption=f'mask ratio: {r:0.2f} | caption: {texts[i]}') for i, (image, r) in
                        enumerate(zip(pil_images, mask_ratio))]
        wandb.log({"Original images v.s. Reconstructed images v.s. Predicted images": wandb_images}, step=global_step)
    except Exception as e:
        logger.warning(f"Skipping prediction image logging at step {global_step}: {e}")

    model.train()


@torch.no_grad()
def generate_images(
        model,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
        mask_schedule,
):
    logger.info("Generating images...")
    model.eval()

    # read validation prompts from file
    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()


    mask_dtype = model.get_input_embeddings().weight.dtype
    mask_token_id = accelerator.unwrap_model(model).config.mask_token_id
    image_tokens = torch.ones((len(validation_prompts), config.model.mmada.num_vq_tokens), dtype=torch.long,
                              device=accelerator.device) * mask_token_id
    input_ids, attention_mask = uni_prompting((validation_prompts, image_tokens), 't2i_gen')
    if config.training.guidance_scale > 0:
        uncond_input_ids, uncond_attention_mask = uni_prompting(([''] * len(validation_prompts), image_tokens), 't2i_gen')
    else:
        uncond_input_ids = None
        uncond_attention_mask = None
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
        # Generate images
        gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            uncond_attention_mask=uncond_attention_mask,
            guidance_scale=config.training.guidance_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=config.training.generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            predict_all_tokens=config.training.get("predict_all_tokens", False),
            seq_len=config.model.mmada.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )
    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids)

    model.train()

    if config.training.get("pre_encode", False):
        del vq_model

    # Convert to PIL images
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    # Log images
    try:
        wandb_images = [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]
        wandb.log({"Generated images": wandb_images}, step=global_step)
    except Exception as e:
        logger.warning(f"Skipping generated image logging at step {global_step}: {e}")
    
    

@torch.no_grad()
def understanding_images(
        model,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
):
    logger.info("Understanding images...")
    model.eval()
        
    file_list = os.listdir(config.dataset.params.mmu_image_root)
    file_list = [f for f in file_list if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    responses = ['' for i in range(len(file_list))]
    images = []
    
    device = accelerator.device
    
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    
    for i, file_name in enumerate(file_list):
        image_path = os.path.join(config.dataset.params.mmu_image_root, file_name)
        image_ori = Image.open(image_path).convert("RGB")
        image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
        image = image.unsqueeze(0)
        images.append(image)
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        batch_size = 1
        
        input_ids = uni_prompting.text_tokenizer(['<|start_header_id|>user<|end_header_id|>\n' + "Please describe this image in detail."  +'<eot_id><|start_header_id|>assistant<|end_header_id|>\n'])['input_ids']
        input_ids = torch.tensor(input_ids).to(device)

        input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
            image_tokens,
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
            input_ids
        ], dim=1).long()
        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
            output_ids = accelerator.unwrap_model(model).mmu_generate(input_ids)
        # output_ids = torch.stack(output_ids).squeeze()[None]

        text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
        responses[i] += text[0]
    model.train()
    images = torch.cat(images, dim=0)
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    # Log images
    try:
        wandb_images = [wandb.Image(image, caption=responses[i]) for i, image in enumerate(pil_images)]
        wandb.log({"Understanding images": wandb_images}, step=global_step)
    except Exception as e:
        logger.warning(f"Skipping understanding image logging at step {global_step}: {e}")


def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))

    # Save full trainer state (model/optimizer/lr_scheduler/rng/scaler) for reproducible resume.
    accelerator.save_state(str(save_path / "training_state"))

    if accelerator.is_main_process:
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
