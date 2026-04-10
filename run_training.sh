#!/bin/bash

LOG_FILE="train_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training..."
echo "Log file: $LOG_FILE"

accelerate launch \
  --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
  --main_process_port=8888 \
  training/train_scmmada.py \
  config=configs/scmmada_stage1.yaml \
  > $LOG_FILE 2>&1

echo "Training finished."