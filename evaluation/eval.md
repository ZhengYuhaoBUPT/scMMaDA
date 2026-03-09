# Evaluation Demo

This is an evaluation demo for MMaDA.

## 1. VLM Evaluation

We use `VLMEvalKit` to evaluate MMaDA's VLM capabilities.

### 1.1 Install Dependencies
```bash
cd evaluation_demo/VLMEvalKit
pip install -e
```

### 1.2 Configure Model Paths
In `VLMEvalKit/vlmeval/config.py`, set the following paths:

```python
mmada = {
    "MMaDA-MixCoT": partial(
        MMaDA, 
        model_path="Gen-Verse/MMaDA-8B-MixCoT",
        tokenizer_path="/Gen-Verse/MMaDA-8B-MixCoT",
        vq_model_path="showlab/magvitv2",
        vq_model_type="magvitv2",
        resolution=512,
    ),
}
```

### 1.3 Configure Dataset Configs
In `VLMEvalKit/vlmeval/vlm/mmada/dataset_configs.py`, you can set the max_new_tokens, steps, block_length for each dataset. For example: 

```python
DATASET_CONFIGS = {
    "MathVista_MINI": {
        "max_new_tokens": 96,
        "steps": 96,
        "block_length": 48,
    },
    
    "MathVerse_MINI_Vision_Only": {
        "max_new_tokens": 256,
        "steps": 128,
        "block_length": 32,
    },
}
```

### 1.4 Run VLM Evaluation
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python run.py --data {dataset_name} --model MMaDA-MixCoT

# Multi-GPU
torchrun --nproc-per-node=8 --master-port=54321 run.py --data {dataset_name} --model MMaDA-MixCoT

# USE COT 
USE_COT=1 torchrun --nproc-per-node=8 --master-port=54321 run.py --data MathVista_MINI --model MMaDA-MixCoT
```

## 2. LLM Evaluation

We directly adopt LLaDA and Fast-dLLM's evaluation scripts. Please note we have not yet implemented and tuned the reasoning process and currently only implemented the non-thinking version, and the results are not yet aligned with our internal results. 
Configuring `lm_eval_harness` in the future may resolve this issue.

### 2.1 Install Dependencies
```bash
cd evaluation_demo/lm
pip install lm-eval 
```

### 2.2 Run LLM Evaluation
```bash
# Using lm-eval-harness
bash eval.sh
```

## 3. Text to image generation

We use [GenEval](https://github.com/djghosh13/geneval) to evaluate the text to image generation capabilities of MMaDA. Please refer to the [GenEval](https://github.com/djghosh13/geneval) for specific instructions.



