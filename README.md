# 使用 Unsloth 微调 Gemma-3N 模型

本项目提供了一套完整的脚本，用于将自定义的指令格式数据集转换为 ShareGPT 格式，并使用 Unsloth 对 Gemma-3N 系列模型进行高效的 LoRA 微调。

## 流程概览

1.  **环境安装**: 安装所有必需的 Python 依赖库。
2.  **数据转换**: 将您自己的 `instruction`/`output` 格式的 JSON 数据集转换为 Unsloth/ShareGPT 兼容的 `.jsonl` 格式。
3.  **模型微调**: 使用转换后的数据集对 Gemma-3N 模型进行微调。

---

## 步骤 1: 环境安装

首先，克隆仓库并安装 `requirements.txt` 中列出的所有依赖项。建议在一个虚拟环境（如 venv 或 conda）中进行安装。

```bash
# 安装依赖
pip install -r requirements.txt
```

---

## 步骤 2: 数据准备与转换

### 2.1 准备您的数据集

您的原始数据集需要是一个 JSON 文件，其中包含一个对象列表，每个对象都有 `instruction` 和 `output` 两个键。

**示例 `my_data.json`:**
```json
[
    {
        "instruction": "你好，请介绍一下你自己。",
        "output": "我是一个大型语言模型，由 Unsloth 训练。"
    },
    {
        "instruction": "解释一下光合作用。",
        "output": "光合作用是植物利用光能，将二氧化碳和水转化为能量（葡萄糖）并释放氧气的过程。"
    }
]
```
将您的数据集文件（例如 `dataset/NekoQA-10K.json`）放置在项目目录中。

### 2.2 运行转换脚本

使用 `dataset/convert_dataset.py` 脚本将您的数据转换为 `.jsonl` 格式。

```bash
python dataset/convert_dataset.py \
    --input_file dataset/NekoQA-10K.json \
    --output_file dataset/NekoQA-10K_converted.jsonl
```
*   `--input_file`: 您原始的 JSON 数据文件路径。
*   `--output_file`: 转换后输出的 `.jsonl` 文件路径。

该脚本会生成一个 `NekoQA-10K_converted.jsonl` 文件，每一行都是一个符合 ShareGPT 格式的 JSON 对象。

---

## 步骤 3: 模型微调

准备好转换后的数据集后，运行 `train_gemma3n.py` 脚本来开始微调。

```bash
python train_gemma3n.py \
    --dataset_path dataset/NekoQA-10K_converted.jsonl \
    --output_dir gemma-3n-neko-finetuned
```

### 主要命令行参数

您可以通过命令行参数自定义训练过程：

*   `--dataset_path`: **[必需]** 转换后的 `.jsonl` 数据集文件路径。
*   `--output_dir`: **[可选]** 保存微调后 LoRA 适配器的目录。默认为 `gemma-3n-finetuned`。
*   `--model_name`: **[可选]** 要使用的基础模型。默认为 `unsloth/gemma-3n-E4B-it`。您可以从 [Unsloth Hugging Face](https://huggingface.co/unsloth) 选择其他模型。
*   `--num_train_epochs`: **[可选]** 训练的轮数。默认为 `1`。
*   `--max_steps`: **[可选]** 最大训练步数。如果设置，将覆盖 `--num_train_epochs`。
*   `--learning_rate`: **[可选]** 学习率。默认为 `2e-4`。
*   `--max_seq_length`: **[可选]** 模型处理的最大序列长度。默认为 `1024`。

训练完成后，微调好的 LoRA 适配器和分词器配置文件将保存在您指定的输出目录中。