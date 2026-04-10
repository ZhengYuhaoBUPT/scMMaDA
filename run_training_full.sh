#!/bin/bash
# Set CUDA_HOME before running
export CUDA_HOME=/usr/local/cuda-12.8

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zyh

# Run accelerate with the correct configuration
# Using GPU 2,3,4,5 as configured
accelerate launch --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml --main_process_port=8888 training/train_scmmada.py config=configs/mmada_pretraining_stage1_llada_instruct.yaml
