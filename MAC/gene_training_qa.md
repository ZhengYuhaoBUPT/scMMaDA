# 基因训练格式与两阶段训练问答记录

## 1. 项目当前训练主线

用户问：当前目录是我们的代码，阅读完整项目后，想确认我们训练的数据是不是类似 `mmu` 格式；我们参考的是 MMaDA。还想知道基因数据的 caption 是什么，以及两阶段训练该怎么写进 README。

结论：

- 这个仓库里目前有两条线并存：
  - 原始 MMaDA 的 image/text/mmu 训练线
  - 你们新增的 gene 分支：`train_scmmada.py + CellwTextDataset + <|mmug|>`
- 公开的 `README.md` 仍然主要是原始 MMaDA 说明，不是你们当前 gene 训练逻辑的完整说明。
- 基因分支设计上是“类似 mmu 的模板”，但把图像 token 换成了 gene token。

## 2. 基因训练模板格式

用户问：我们代码里基因训练模板格式包括什么占位符？

回答：

你们代码里，基因训练用的模板按设计是：

```text
<|mmug|> <|soi|> gene_tokens <|eoi|> text_tokens
```

也就是：

- `<|mmug|>`：gene understanding 任务 token
- `<|soi|>`：模态内容开始
- `gene_tokens`：基因 token 序列
- `<|eoi|>`：模态内容结束
- 后面接文本 token

你们代码里目前没有单独的：

- `<|sog|>`
- `<|eog|>`

而是直接复用了：

- `<|soi|>`
- `<|eoi|>`

所以用户提出的：

```text
<mmug><sog>基因token<eog><sot>文本描述
```

在“语义上”是对的，但在“代码实现上”实际写法更接近：

```text
<|mmug|><|soi|>基因token<|eoi|>文本
```

## 3. 原始 MMaDA 有没有 caption

用户问：MMaDA 是有 caption 的对吗，我们没有对吗？

回答：

对。

原始 MMaDA 有 caption，而且是训练主干之一：

- `t2i` 用图文对做生成
- `mmu` 用 image caption / image instruction 做理解
- 在图像数据加载里会读取外部 caption

而你们当前 gene 分支按代码实现看，没有真正接入 caption：

- `CellwTextDataset` 只返回：
  - `gene_ids`
  - `gene_expression`
  - `celltype_label`
- 没有返回 `texts` / `caption`
- 训练时如果拿不到 `texts`，就会默认用空字符串

所以可以概括为：

- 原始 MMaDA 有 caption
- 你们当前 gene 分支默认没有真正接入 caption，当前更像“gene token 输入接口已经有了，但文本监督目标还没完整接好”

## 4. gene_expression 和 celltype_label 是不是 caption

用户问：我们的 `gene_expression`、`celltype_label` 是 caption 吗？

回答：

不是。

它们的角色分别更接近：

- `gene_expression`：数值或离散表达信息，是输入特征
- `celltype_label`：结构化类别标签，是 annotation
- `caption`：自然语言文本监督目标

例如一条样本如果有：

- `gene_ids = [CD3D, IL7R, LTB, ...]`
- `gene_expression = [高, 高, 中, ...]`
- `celltype_label = "CD4 naive T cell"`

那么：

- `gene_expression` 不是 caption
- `celltype_label` 也不是 caption
- 但你可以把这些信息写成自然语言，生成 caption，比如：

```text
This cell is a CD4 naive T cell.
```

或者：

```text
This cell is a CD4 naive T cell with high expression of IL7R and LTB.
```

这些自然语言文本才算 caption / text target。

## 5. celltype、disease、tissue 能不能作为 caption

用户问：我们数据集中包括 `celltype`、`disease`、`tissue`，比如：

- `celltype = natural killer cell`
- `disease = normal`
- `tissue = cortex of kidney`

这些可以作为 caption 吗？

回答：

可以，但更准确地说：

- 它们原本是结构化 metadata / label
- 把它们组织成自然语言之后，就可以作为 caption

例如下面这些都可以作为第一阶段的文本目标：

```text
This cell is a natural killer cell.
```

```text
This cell is from the cortex of kidney.
```

```text
This cell is a natural killer cell from the cortex of kidney under normal condition.
```

所以对你们项目，更准确的说法可以是：

- `gene-to-text description`
- `cell annotation text`

而不一定非要叫 image-style caption。

如果放回模板里，就是：

```text
<|mmug|><|soi|> gene_tokens <|eoi|> This cell is a natural killer cell from the cortex of kidney under normal condition.
```

## 6. 两阶段训练应该怎么理解

用户问：如果我们训练是两个阶段，README 应该怎么写？第一阶段训练数据怎么说，第二阶段 CoT 怎么说？

回答：

### 第一阶段：基础对齐 / 非推理训练

目标：

- 先学会 gene token 到自然语言描述的映射
- 不强调推理过程

第一阶段的数据建议写成：

```text
<|mmug|><|soi|> gene_tokens <|eoi|><|start_header_id|>user<|end_header_id|>
Describe this cell based on its gene expression profile.
<eot_id><|start_header_id|>assistant<|end_header_id|>
This cell is a natural killer cell from the cortex of kidney under normal condition.
```

这一阶段的 target 更像：

- cell type description
- tissue description
- disease status description
- marker summary

所以第一阶段更适合写成：

- gene-to-text description learning
- cell annotation alignment

而不是复杂 QA 或 CoT。

### 第二阶段：CoT / 推理增强训练

目标：

- 在第一阶段基础上，继续训练“问题 + 推理链 + 答案”
- 让模型学会基于 gene 表达进行 reasoning

第二阶段的数据建议写成：

```text
<|mmug|><|soi|> gene_tokens <|eoi|><|start_header_id|>user<|end_header_id|>
You should first think about the reasoning process in the mind and then provide the answer. The reasoning process is enclosed within <think> </think> tags.
What is the most likely cell type of this cell?
<eot_id><|start_header_id|>assistant<|end_header_id|>
<think>The cell shows marker genes consistent with NK lineage and does not match T-cell or myeloid signatures.</think>
The cell is most likely a natural killer cell.
```

所以可以概括为：

- 第一阶段学“描述”
- 第二阶段学“推理”

也就是：

- Stage 1: gene-to-text description / annotation learning
- Stage 2: gene-conditioned QA with CoT

## 7. 一个关键实现问题

在代码检查中还发现一个当前 gene 分支的重要问题：

- `CellwTextDataset` 没有输出 `texts`
- 训练时 gene 分支默认会回退到空文本

另外，`mmug` 调用时参数顺序也存在不一致：

- 设计上应该是 `(texts, gene_ids)`
- 训练里当前传的是 `(gene_ids, texts_mmug)`

所以当前 `mmug` 训练分支从实现角度看仍然是不完整的，需要后续修正。

## 8. 对当前项目最准确的一句话总结

当前项目里：

- 原始 MMaDA 部分是有 caption 的
- 你们新增的 gene 分支在设计上已经定义了 `<|mmug|><|soi|>gene_tokens<|eoi|>text`
- 但当前 gene 数据并没有真正完整接入对应的自然语言 caption
- 如果要做两阶段训练，第一阶段建议使用 metadata 组织成自然语言描述，第二阶段再使用 question + CoT + answer
