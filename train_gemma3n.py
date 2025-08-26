#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_nekoqa_lora_stable_v2.py

主要改动：
- ✅ 多轮对话“炸成单轮样本”，仅对每个 assistant 回复计 loss（掩码更稳）
- ✅ tokenizer 左截断（不切掉答案），左填充
- ✅ LoRA 容量下调 + RSLoRA + NEFTune + 正则/标签平滑，降低过拟合
- ✅ 切出 10% 验证集 + 早停（不会触发 Dynamo 的重编译问题）
- ✅ 默认仅训注意力层（必要时再加 MLP）
- ✅ 推理超参更稳（轻微降复读）

用法示例：
python train_nekoqa_lora_stable_v2.py \
  --dataset_path dataset/NekoQA-10K_convert.jsonl \
  --output_dir gemma3n-neko-lora \
  --merged_dir gemma3n-neko-lora-merged \
  --max_steps 1000 --per_device_train_batch_size 4 --gradient_accumulation_steps 4
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Unsloth 必须最先导入
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import argparse
import random
import torch
from datasets import load_dataset
from typing import List, Dict

from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

# ===== 默认可调参数 =====
BASE_MODEL = "unsloth/gemma-3n-E4B-it"
CHAT_TEMPLATE_NAME = "gemma-3"
DATASET_PATH = "dataset/NekoQA-10K_convert.jsonl"
OUTPUT_DIR = "gemma3n-neko-lora"
MERGED_DIR = "gemma3n-neko-lora-merged"

# LoRA & 训练
LORA_R = 16
LORA_ALPHA = 48
LORA_DROPOUT = 0.10
LEARNING_RATE = 1e-4
MAX_STEPS = 2000
PER_DEVICE_TRAIN_BSZ = 4
GRAD_ACC_STEPS = 4
MAX_SEQ_LEN = 1024
USE_4BIT = True
SEED = 3407
DO_MERGE = True
VALID_RATIO = 0.1

WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
WARMUP_RATIO = 0.10

# ===== 工具函数 =====
def detect_assistant_prefix(tokenizer) -> str:
    """探测“assistant 段开始之前”的前缀，用于掩码"""
    sentinel = "<|ASSISTANT_CONTENT|>"
    probe = [
        {"role": "user", "content": "DUMMY"},
        {"role": "assistant", "content": sentinel},
    ]
    rendered = tokenizer.apply_chat_template(
        probe, tokenize=False, add_generation_prompt=False
    )
    idx = rendered.find(sentinel)
    if idx == -1:
        raise RuntimeError("无法探测 assistant 前缀：未找到 sentinel。")
    start_token = "<start_of_turn>"
    j = rendered.rfind(start_token, 0, idx)
    if j != -1:
        return rendered[j:idx]
    k = rendered.rfind("\n", 0, idx)
    return rendered[k+1:idx] if k != -1 else rendered[:idx]

def render_text(tokenizer, msgs: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )

# ===== 自定义 collator：仅在“单轮样本”的 assistant 段计算 loss =====
class AssistantOnlyCollator:
    """
    前置条件：样本是“单轮样本”（历史 user 若干 + 当前 assistant 一段）。
    实现：把 assistant 段开始之前（含前缀）的 token 都置为 -100。
    """
    def __init__(self, tokenizer, assistant_prefix: str, max_seq_len: int):
        self.tokenizer = tokenizer
        self.ap = assistant_prefix
        self.max_len = max_seq_len

    def __call__(self, batch):
        texts = [x["text"] for x in batch]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )
        labels = enc["input_ids"].clone()

        for i, t in enumerate(texts):
            pos = t.find(self.ap)
            if pos == -1:
                # 未找到前缀，整条不计 loss
                labels[i, :] = -100
                continue
            prefix = t[: pos + len(self.ap)]
            pref_ids = self.tokenizer(
                prefix, return_tensors="pt",
                truncation=True, max_length=self.max_len,
            )["input_ids"][0]
            pref_len = pref_ids.shape[0]
            labels[i, :pref_len] = -100

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

# ===== 模型 & tokenizer & 生成 =====
def load_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=USE_4BIT,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    # LoRA：先只训注意力层，容量下调 + 正则
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],  # 先不动 MLP
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=SEED,
        use_rslora=True,              # RSLoRA 稳定 rank
        loftq_config=None,
        neftune_noise_alpha=5,        # 轻量“噪声微调”，抑制过拟合
    )

    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"   # ✅ 保留结尾（答案）
    return model, tokenizer

def quick_generate_no_system(model, tokenizer, user_text):
    msgs = [{"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text=[prompt], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True, temperature=0.8, top_p=0.9,
            repetition_penalty=1.08,  # 略降复读
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# ===== 数据处理 =====
def explode_to_single_turn(example):
    """
    把一条多轮对话炸成多条“单轮样本”：
    [u1, a1, u2, a2, ...] -> 两条样本：([u1, a1]), ([u1, a1, u2, a2])
    实际实现：每遇到一个 assistant，就把“到该 assistant 为止”的对话作为一条样本。
    """
    msgs = example["messages"]
    out = []
    history = []
    for m in msgs:
        if m["role"] == "assistant":
            out.append({"messages": history + [m]})
        else:
            history.append(m)
    return {"samples": out}

def flatten_exploded(batch):
    # 将 {"samples":[{"messages":[...]}, ...]} 扁平化为行
    flat = []
    for item in batch["samples"]:
        flat.extend(item)
    return {"messages": [x["messages"] for x in flat]}

# ===== 主流程 =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--merged_dir", type=str, default=MERGED_DIR)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--per_device_train_batch_size", type=int, default=PER_DEVICE_TRAIN_BSZ)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRAD_ACC_STEPS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--do_merge", action="store_true", default=DO_MERGE)
    parser.add_argument("--valid_ratio", type=float, default=VALID_RATIO)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO)
    args = parser.parse_args()

    # 随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f">> 读取数据：{args.dataset_path}")
    raw = load_dataset("json", data_files=args.dataset_path, split="train")

    # 切分验证集
    print(">> 切分训练/验证集 …")
    split = raw.train_test_split(test_size=args.valid_ratio, seed=args.seed)
    ds_train_raw = split["train"]
    ds_valid_raw = split["test"]

    # 炸成单轮样本
    print(">> 多轮对话炸成单轮样本 …")
    ds_train = ds_train_raw.map(explode_to_single_turn)
    ds_train = ds_train.map(flatten_exploded, batched=True, remove_columns=ds_train.column_names)
    ds_valid = ds_valid_raw.map(explode_to_single_turn)
    ds_valid = ds_valid.map(flatten_exploded, batched=True, remove_columns=ds_valid.column_names)

    print(">> 加载模型…")
    model, tokenizer = load_model_and_tokenizer()

    # 训练前示例
    print("\n[训练前] 生成示例：")
    try:
        print(quick_generate_no_system(model, tokenizer, "宝宝，如果我走了，你会怎么做？"), "\n")
    except Exception as e:
        print(f"(示例生成失败，不影响训练) -> {e}\n")

    # 探测 assistant 前缀
    apref = detect_assistant_prefix(tokenizer)
    print(f"[debug] assistant_prefix = {repr(apref[:80])} ...")

    # 渲染为 text
    print(">> 渲染数据为 text 格式 …")
    def _render_to_text(examples):
        return {"text": [render_text(tokenizer, m) for m in examples["messages"]]}

    train_text = ds_train.map(
        _render_to_text,
        batched=True,
        remove_columns=ds_train.column_names,
    )
    valid_text = ds_valid.map(
        _render_to_text,
        batched=True,
        remove_columns=ds_valid.column_names,
    )

    collator = AssistantOnlyCollator(tokenizer, apref, MAX_SEQ_LEN)

    print(">> 配置 SFTTrainer（轻量验证 + 早停）…")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_text,
        eval_dataset=valid_text,
        data_collator=collator,
        dataset_text_field="text",
        args=SFTConfig(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,

            warmup_ratio=args.warmup_ratio,
            max_steps=args.max_steps,
            num_train_epochs=1,            # 被 max_steps 覆盖
            logging_steps=20,

            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=20,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            label_smoothing_factor=args.label_smoothing,
            max_grad_norm=1.0,
            weight_decay=args.weight_decay,

            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            lr_scheduler_type="cosine",
            seed=args.seed,
            packing=False,                 # 与自定义 collator 并用更安全
            max_seq_length=MAX_SEQ_LEN,

            optim="adamw_torch",
            adam_beta2=0.95,
            dataloader_num_workers=0,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n>> 开始训练 …")
    trainer.train()

    print("\n>> 保存 LoRA 适配器 …")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA 保存在: {args.output_dir}")

    if args.do_merge:
        print("\n>> 合并 LoRA 到基座（merge_and_unload） …")
        from transformers import AutoModelForCausalLM
        from peft import PeftModel

        # 重新加载基座（非 4bit），再合并
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
        )
        peft = PeftModel.from_pretrained(base, args.output_dir)
        merged = peft.merge_and_unload()
        os.makedirs(args.merged_dir, exist_ok=True)
        merged.save_pretrained(args.merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.merged_dir)
        print(f"合并完成：{args.merged_dir}")

    print("\n[训练后] 生成示例：")
    try:
        print(quick_generate_no_system(model, tokenizer, "早晨如何与主人互动？"))
    except Exception as e:
        print(f"(示例生成失败，不影响训练) -> {e}")
    print(f"\n✅ 完成！LoRA: {args.output_dir}" + (f"；合并模型: {args.merged_dir}" if args.do_merge else ""))


if __name__ == "__main__":
    main()
