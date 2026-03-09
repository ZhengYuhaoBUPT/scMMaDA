# MMaDA 项目入门指南

## 项目概述

MMaDA (Multimodal Large Diffusion Language Models) 是一个统一的多模态扩散基础模型家族，在文本推理、多模态理解和文本到图像生成等多个领域表现出色。

### 核心创新点

1. **统一扩散架构** - 采用共享概率公式和模态无关设计，消除模态特定组件
2. **混合长链式思维 (Mix-CoT) 微调** - 跨模态统一思维链格式
3. **UniGRPO 强化学习算法** - 基于策略梯度的统一 RL，用于扩散基础模型

## 项目结构

```
MMaDA/
├── models/                      # 模型定义
│   ├── modeling_mmada.py         # MMaDA 核心模型
│   ├── modeling_llada.py        # LLaDA 基础模型
│   ├── modeling_magvitv2.py     # MAGVIT V2 VQ 模型
│   ├── configuration_llada.py    # 配置类
│   └── sampling.py             # 采样方法
│
├── training/                   # 训练脚本
│   ├── train_mmada.py          # 阶段1训练 (ImageNet)
│   ├── train_mmada_stage2.py   # 阶段2训练 (Image-Text)
│   ├── train_mmada_stage3.py   # 阶段3训练 (Text Instruction)
│   ├── train_mmada_cot_sft.py  # 阶段4训练 (Mix-CoT)
│   ├── train_mmada_stage4.py   # 阶段5训练 (MultiModal CoT)
│   ├── data.py                # 数据加载
│   ├── prompting_utils.py      # 提示工程工具
│   └── utils.py              # 工具函数
│
├── configs/                    # 配置文件
│   ├── mmada_demo.yaml                        # 演示配置
│   ├── mmada_pretraining_stage1_llada_instruct.yaml
│   ├── mmada_pretraining_stage2_llada_instruct.yaml
│   ├── mmada_pretraining_stage3_llada_instruct.yaml
│   ├── mmada_pretraining_stage3_llada_instruct_512_cot.yaml
│   └── mmada_pretraining_stage4_llada_instruct.yaml
│
├── accelerate_configs/          # Accelerate 分布式训练配置
│   ├── 1_gpu.yaml
│   ├── 1_node_8_gpus_deepspeed_zero2.yaml
│   └── 8_node_8_gpus_deepspeed_zero2.yaml
│
├── evaluation/                 # 评估工具
│   └── VLMEvalKit/           # 视觉语言模型评估套件
│
├── app.py                     # Gradio 演示应用
├── generate.py                # 文本生成推理
├── inference_mmu.py           # 多模态生成推理
├── inference_t2i.py           # 文本到图像推理
└── requirements.txt           # 依赖包
```

## 快速开始

### 1. 环境安装

```bash
# 安装依赖
pip install -r requirements.txt
```

主要依赖包括：
- `transformers==4.46.0` - 模型框架
- `diffusers==0.32.2` - 扩散模型
- `accelerate` - 分布式训练
- `deepspeed` - 大规模训练
- `gradio>=4.44.1` - Web 演示界面
- `wandb` - 实验跟踪

### 2. 运行演示应用

```bash
# 启动本地 Gradio 演示
python app.py
```

演示应用包含三个功能：
- **Part 1: 文本生成** - 纯文本推理和生成
- **Part 2: 多模态理解** - 图像描述和问答
- **Part 3: 文本到图像生成** - 从文本生成图像

### 3. 批量推理

#### 文本生成
```bash
python generate.py
```

#### 多模态生成
```bash
# 首先登录 wandb
wandb login

python3 inference_mmu.py \
  config=configs/mmada_demo.yaml \
  mmu_image_root=./mmu_validation \
  mmu_prompts_file=./mmu_validation/prompts_with_vqa.json
```

#### 文本到图像生成
```bash
python3 inference_t2i.py \
  config=configs/mmada_demo.yaml \
  batch_size=1 \
  validation_prompts_file=validation_prompts/text2image_prompts.txt \
  guidance_scale=3.5 \
  generation_timesteps=15
```

## 模型架构

### MMadaModelLM (models/modeling_mmada.py)

核心模型类，继承自 `LLaDAModelLM`，提供以下关键功能：

#### 1. 文本生成方法
- `mmu_generate()` - 多模态/文本生成的扩散采样
- `mmu_generate_fast()` - 快速生成版本（支持提前停止）

**核心参数：**
- `max_new_tokens`: 最大生成长度
- `steps`: 扩散步数
- `block_length`: 分块长度
- `temperature`: 控制随机性
- `cfg_scale`: 分类器自由引导强度
- `remasking`: 重掩码策略 ('low_confidence' 或 'random')

#### 2. 图像生成方法
- `t2i_generate()` - 文本到图像生成
- `t2i_generate_decoding_stepwise()` - 逐步解码版本（用于可视化）

**采样过程：**
1. 初始化图像 token 为全 mask
2. 通过多步迭代逐步去噪
3. 每步预测部分 token，基于置信度选择保留
4. 使用余弦噪声调度控制 mask 比例

#### 3. 训练前向方法
- `forward_process()` - 标准 T2I + LM + MMU 训练
- `forward_process_with_r2i()` - 包含 reasoning-to-image 任务
- `forward_t2i()` - 仅 T2I 训练

### 关键组件

1. **Gumbel 噪声采样** (add_gumbel_noise)
   - 使用 float64 精度提高质量
   - 支持温度参数控制随机性

2. **掩码调度** (get_num_transfer_tokens)
   - 预计算每步需要转换的 token 数量
   - 线性噪声调度确保一致性

3. **注意力偏置**
   - 支持 T2I 任务的特定注意力模式
   - 通过 attention_bias 控制可见性

## 训练流程

### 阶段 1: ImageNet 预训练

在 ImageNet 上训练获得基本视觉能力：

```bash
accelerate launch --config_file path/to/accelerate_config \
  --main_process_port=8888 \
  training/train_mmada.py \
  config=configs/mmada_pretraining_stage1_llada_instruct.yaml
```

**配置要点：**
- 使用 LLaDA-8B-Instruct 初始化
- 在 ImageNet 上训练
- 目标：学习图像-文本对齐

### 阶段 2: 图像-文本数据预训练

替换 ImageNet 为图像-文本数据集：

```bash
accelerate launch --config_file path/to/accelerate_config \
  --main_process_port=8888 \
  training/train_mmada_stage2.py \
  config=configs/mmada_pretraining_stage2_llada_instruct.yaml
```

**注意：** 需更新配置中的 `pretrained_model_path` 指向阶段1的检查点

### 阶段 3: 文本指令微调

训练文本指令跟随能力：

```bash
accelerate launch --config_file path/to/accelerate_config \
  --main_process_port=8888 \
  training/train_mmada_stage3.py \
  config=configs/mmada_pretraining_stage3_llada_instruct.yaml
```

### 阶段 4.1: Mix-CoT 文本推理训练

纯文本推理的链式思维训练：

```bash
accelerate launch --config_file path/to/accelerate_config \
  --main_process_port=8888 \
  training/train_mmada_cot_sft.py \
  config=configs/mmada_pretraining_stage3_llada_instruct_512_cot.yaml
```

**思维链格式：**
```
<thinking>
推理过程...
</thinking>
答案...
```

### 阶段 4.2: 多模态 Mix-CoT 训练

加入多模态推理能力：

```bash
accelerate launch --config_file path/to/accelerate_config \
  --main_process_port=8888 \
  training/train_mmada_stage4.py \
  config=configs/mmada_pretraining_stage4_llada_instruct.yaml
```

### 阶段 5: UniGRPO 强化学习

使用策略梯度算法进一步优化（推荐使用 dLLM-RL 框架）：

详情参考：[dLLM-RL](https://github.com/Gen-Verse/dLLM-RL)

## 配置文件说明

配置文件使用 YAML 格式，主要部分：

### model 部分
```yaml
model:
  vq_model:
    type: "magvitv2"
    vq_model_name: "showlab/magvitv2"

  mmada:
    pretrained_model_path: "Gen-Verse/MMaDA-8B-Base"
    new_vocab_size: 134656
    llm_vocab_size: 126464
    codebook_size: 8192
    num_vq_tokens: 1024
    num_new_special_tokens: 0
```

### dataset 部分
```yaml
dataset:
  gen_type: "imagenet1k"
  und_type: "captioning"
  combined_loader_mode: "max_size_cycle"
  params:
    train_t2i_shards_path_or_url: "path/to/imagenet"
    train_mmu_shards_path_or_url: ["path/to/sa-1b", "path/to/cc12m"]
    train_lm_shards_path_or_url: "path/to/text-data"
```

### training 部分
```yaml
training:
  batch_size_t2i: 5
  batch_size_lm: 1
  batch_size_mmu: 2
  mixed_precision: "bf16"
  guidance_scale: 1.5
  generation_timesteps: 12
  t2i_coeff: 1.0
  lm_coeff: 0.1
  mmu_coeff: 1.0
```

## Accelerate 配置

准备加速配置：

```bash
# 交互式配置
accelerate config

# 或使用预设配置
# 单 GPU
accelerate_configs/1_gpu.yaml

# 8 卡节点 + DeepSpeed Zero-2
accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml

# 8 节点 x 8 GPU
accelerate_configs/8_node_8_gpus_deepspeed_zero2.yaml
```

## 数据处理

### 特殊 Token
项目使用以下特殊 token 用于模态分隔：
- `<|soi|>` - 图像开始
- `<|eoi|>` - 图像结束
- `<|sov|>` - 视频开始
- `<|eov|>` - 视频结束
- `<|t2i|>` - 文本到图像
- `<|mmu|>` - 多模态理解
- `<|t2v|>` - 文本到视频
- `<|v2v|>` - 视频到视频
- `<|lvg|>` - 长视频生成

### UniversalPrompting (training/prompting_utils.py)

统一的提示格式处理类，支持：
- 模态编码/解码
- 条件丢弃（用于 CFG）
- 特殊 token 管理

## 推理参数调优

### 文本生成

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| steps | 256-512 | 扩散步数，越多越慢但质量可能更好 |
| gen_length | 256-512 | 生成长度 |
| block_length | 64-128 | 分块长度 |
| temperature | 0.7-1.2 | 温度，0 为确定性 |
| cfg_scale | 0.0-0.5 | CFG 强度，文本生成通常较低 |
| remasking | 'low_confidence' | 掩码策略 |

### 图像生成

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| steps | 15-20 | 扩散步数 |
| guidance_scale | 3.0-5.0 | 引导强度 |
| scheduler | 'cosine' | 噪声调度 |
| resolution | 512 | 图像分辨率 |

## 评估

参考 `evaluation/eval.md` 了解详细的评估方法。

## 可用模型

- **MMaDA-8B-Base**: 预训练和指令微调后的基础模型
- **MMaDA-8B-MixCoT**: Mix-CoT 微调模型，支持复杂推理
- **MMaDA-Parallel-M**: 并行思维感知模型

## 常见问题

### Q1: 如何修改数据路径？
在对应的 `configs/xxx.yaml` 文件中修改 `dataset.params` 下的路径。

### Q2: 如何调整训练批次大小？
修改配置文件中 `training` 部分的 `batch_size_t2i`, `batch_size_lm`, `batch_size_mmu`。

### Q3: 推理时显存不足怎么办？
- 减小 `gen_length` 或 `block_length`
- 减小 `steps`
- 使用 `model.to(torch.float16)` 降低精度

### Q4: 如何启用 Thinking Mode？
在 app.py 中点击 "🧠 Enable Thinking Mode" 按钮，或在代码中设置 `thinking_mode_lm=True`。

## 参考资料

- 论文: [MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809)
- Hugging Face: [Gen-Verse/MMaDA](https://huggingface.co/Gen-Verse)
- RL 框架: [dLLM-RL](https://github.com/Gen-Verse/dLLM-RL)
- Demo: [Hugging Face Space](https://huggingface.co/spaces/Gen-Verse/MMaDA)

## 引用

```bibtex
@article{yang2025mmada,
  title={MMaDA: Multimodal Large Diffusion Language Models},
  author={Yang, Ling and Tian, Ye and Li, Bowen and Zhang, Xinchen and Shen, Ke and Tong, Yunhai and Wang, Mengdi},
  journal={arXiv preprint arXiv:2505.15809},
  year={2025}
}
```
