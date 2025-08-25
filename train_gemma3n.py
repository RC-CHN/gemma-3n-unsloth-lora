#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_nekoqa_lora.py
正式 LoRA 训练（Unsloth + Gemma-3N）：
- 读取 JSONL（每行 {"messages":[...]}）
- 使用 chat_template 渲染为纯文本
- 自定义 collator：仅在 assistant 段计算 loss
- 训练中关闭 eval，避免 TorchDynamo 重编译上限错误
- 可选一键 merge LoRA -> 合并模型

默认数据文件：dataset/NekoQA-10K_convert.jsonl
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Unsloth 要最先导入，确保加速补丁生效
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import argparse
import json
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ========== 自定义 collator：只在 assistant 段计算 loss ==========
class AssistantOnlyCollator:
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
                labels[i, :] = -100
                continue
            prefix = t[: pos + len(self.ap)]
            pref_ids = self.tokenizer(
                prefix, return_tensors="pt",
                truncation=True, max_length=self.max_len,
            )["input_ids"][0]
            pref_len = pref_ids.shape[0]
            labels[i, :pref_len] = -100  # 只学习 assistant 段
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

# ========== 辅助：探测 assistant 段的模板前缀 ==========
def detect_assistant_prefix(tokenizer):
    sentinel = "<|ASSISTANT_CONTENT|>"
    probe_msgs = [
        {"role": "user", "content": "DUMMY"},
        {"role": "assistant", "content": sentinel},
    ]
    rendered = tokenizer.apply_chat_template(
        probe_msgs, tokenize=False, add_generation_prompt=False
    )
    idx = rendered.find(sentinel)
    if idx == -1:
        raise RuntimeError("无法探测 assistant 前缀：渲染结果里没有 sentinel。")

    start_token = "<start_of_turn>"
    j = rendered.rfind(start_token, 0, idx)
    if j != -1:
        assistant_prefix = rendered[j:idx]
    else:
        k = rendered.rfind("\n", 0, idx)
        assistant_prefix = rendered[k+1:idx] if k != -1 else rendered[:idx]

    print(f"[debug] assistant_prefix = {repr(assistant_prefix[:120])} ...")
    return assistant_prefix

# ========== 主流程 ==========
def main():
    ap = argparse.ArgumentParser()
    # 数据 & 模型
    ap.add_argument("--dataset_path", type=str, default="dataset/NekoQA-10K_convert.jsonl")
    ap.add_argument("--base_model", type=str, default="unsloth/gemma-3n-E4B-it")
    ap.add_argument("--chat_template", type=str, default="gemma-3")
    ap.add_argument("--output_dir", type=str, default="nekoqa-lora-out")
    ap.add_argument("--merged_dir", type=str, default="nekoqa-gemma3n-merged")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--load_in_4bit", action="store_true", default=True)

    # 训练超参
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=-1)   # >0 则覆盖 epochs
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3407)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.0)  # 大数据建议 0.0 更快

    # 其他
    ap.add_argument("--merge_lora", action="store_true", default=False,
                    help="训练后是否 merge_and_unload 生成合并模型")

    args = ap.parse_args()

    print(f">> 读取数据：{args.dataset_path}")
    ds = load_dataset("json", data_files=args.dataset_path, split="train")
    # 期望每条样本形如 {"messages":[{"role":"user","content":...},{"role":"assistant","content":...}]}

    print(">> 加载模型/分词器 …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    # 训练阶段禁用缓存，减少编译/图切换干扰
    if hasattr(model, "config"):
        model.config.use_cache = False

    # 挂 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # 套 chat 模板（推理/渲染时用）
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # 渲染为纯文本
    print(">> 渲染文本 …")
    def format_batch(examples):
        msgs_list = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in msgs_list
        ]
        return {"text": texts}

    ds_text = ds.map(format_batch, batched=True, remove_columns=[c for c in ds.column_names if c != "messages"])
    # 有的 datasets 版本不允许 remove_columns 移除所有列，这里保留 "messages"
    ds_text = ds_text.remove_columns([c for c in ds_text.column_names if c not in ("messages","text")])

    # 探测 assistant 段前缀 & collator
    apref = detect_assistant_prefix(tokenizer)
    collator = AssistantOnlyCollator(tokenizer, apref, args.max_seq_len)

    # 训练器配置（关闭 eval）
    print(">> 配置 SFTTrainer …")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_text,
        data_collator=collator,
        args=SFTConfig(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            num_train_epochs=args.num_train_epochs,
            logging_steps=args.logging_steps,

            eval_strategy="no",
            load_best_model_at_end=False,

            save_strategy="steps",
            save_steps=args.save_steps,
            output_dir=args.output_dir,

            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            lr_scheduler_type="cosine",
            weight_decay=0.0,
            seed=args.seed,
            packing=False,
            max_seq_length=args.max_seq_len,
        ),
        dataset_text_field="text",
    )

    print("\n==> 开始训练 …")
    trainer.train()

    print("\n>> 保存 LoRA 适配器 …")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA 已保存至：{args.output_dir}")

    if args.merge_lora:
        print("\n>> 合并 LoRA 到基座（merge_and_unload） …")
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        peft = PeftModel.from_pretrained(base, args.output_dir)
        merged = peft.merge_and_unload()
        os.makedirs(args.merged_dir, exist_ok=True)
        merged.save_pretrained(args.merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.merged_dir)
        print(f"合并完成：{args.merged_dir}")

    print("\n✅ 训练完成。")


if __name__ == "__main__":
    main()
