#!/bin/bash

LOG_FILE="train_stage2_$(date +%Y%m%d_%H%M%S).log"

echo "Starting stage2 training..."
echo "Log file: $LOG_FILE"

# accelerate launch \
#   --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2_ours.yaml \
#   --main_process_port=8888 \
#   training/train_scmmada.py \
#   config=configs/scmmada_stage2_ours.yaml \
#   > $LOG_FILE 2>&1

accelerate launch --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2_ours.yaml --main_process_port=8888 training/train_scmmada.py config=configs/scmmada_stage2_ours.yaml

echo "Stage2 training finished."
