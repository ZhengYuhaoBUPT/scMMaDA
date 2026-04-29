
## 🔧 Training
**Update your training data path in `configs/xx.yaml`.**

### Stage 0. Prepare your accelerate configs
Please first prepare your accelerate configs. You can simple run
```
accelerate config
```

Or use our provided configs in `accelerate_configs`:
```
├── accelerate_configs/ 
|   ├── 1_gpu.yaml
|   └── 8_node_8_gpus_deepspeed_zero2.yaml (for 8 * 8 gpus)
```

### Stage 1: Pre-training 
First we use LLaDA-8B-Instruct to initialize our model. 
```
accelerate launch --config_file path/to/your/accelerate_config --main_process_port=8888 training/train_scmmada.py config=configs/scmmada_stage1_ours.yaml
```

### Stage 2
```
accelerate launch --config_file path/to/your/accelerate_config --main_process_port=8888 training/train_scmmada.py config=configs/scmmada_stag21_ours.yaml
```






