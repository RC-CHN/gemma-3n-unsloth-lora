#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_nekoqa_lora_stable.py
- 读取 dataset/NekoQA-10K_convert.jsonl（每行 {"messages":[...]}，建议无 system）
- 动态 collator + 仅在 assistant 段计算 loss（把前缀打 -100）
- 训练中关闭评估（避开 TorchDynamo 重编译）
- 可选训练后 merge LoRA
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Unsloth 必须最先导入
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import argparse
import torch
from datasets import load_dataset, Dataset
from typing import List, Dict

# ===== 可调参数 =====
BASE_MODEL = "unsloth/gemma-3n-E4B-it"
CHAT_TEMPLATE_NAME = "gemma-3"
DATASET_PATH = "dataset/NekoQA-10K_convert.jsonl"
OUTPUT_DIR = "gemma3n-neko-lora"
MERGED_DIR = "gemma3n-neko-lora-merged"

# LoRA & 训练
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
MAX_STEPS = 2000
PER_DEVICE_TRAIN_BSZ = 4
GRAD_ACC_STEPS = 4
MAX_SEQ_LEN = 1024
USE_4BIT = True
SEED = 3407
DO_MERGE = True

# ===== 工具函数 =====
def detect_assistant_prefix(tokenizer) -> str:
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

# ===== 自定义 collator：仅在 assistant 段计算 loss (from mini_batch.py) =====
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

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer

def quick_generate_no_system(model, tokenizer, user_text):
    msgs = [{"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    # ⭐ 同样使用 text=[...] 关键字参数
    inputs = tokenizer(text=[prompt], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True, temperature=0.8, top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# ===== 主流程 =====
def main():
    import random
    from trl import SFTTrainer, SFTConfig

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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f">> 读取数据：{args.dataset_path}")
    ds = load_dataset("json", data_files=args.dataset_path, split="train")

    print(">> 加载模型…")
    model, tokenizer = load_model_and_tokenizer()

    print("\n[训练前] 生成示例：")
    print(quick_generate_no_system(model, tokenizer, "宝宝，如果我走了，你会怎么做？"), "\n")

    apref = detect_assistant_prefix(tokenizer)
    print(f"[debug] assistant_prefix = {repr(apref[:80])} ...")

    print(">> 渲染数据为 text 格式…")
    def _render_to_text(examples):
        return {"text": [render_text(tokenizer, m) for m in examples["messages"]]}

    train_text = ds.map(
        _render_to_text,
        batched=True,
        remove_columns=ds.column_names,
    )

    collator = AssistantOnlyCollator(tokenizer, apref, MAX_SEQ_LEN)

    print(">> 配置 SFTTrainer（无评估）…")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_text,
        data_collator=collator,
        dataset_text_field="text",
        args=SFTConfig(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=100,
            max_steps=args.max_steps,
            num_train_epochs=1,            # 被 max_steps 覆盖
            logging_steps=20,

            eval_strategy="no",
            load_best_model_at_end=False,

            save_strategy="steps",
            save_steps=args.max_steps,     # 结束时存一次
            output_dir=args.output_dir,

            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            lr_scheduler_type="cosine",
            weight_decay=0.0,
            seed=args.seed,
            packing=False,
            max_seq_length=MAX_SEQ_LEN,
        ),
    )

    print("\n>> 开始训练 …")
    trainer.train()

    print("\n>> 保存 LoRA 适配器 …")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.do_merge:
        print("\n>> 合并 LoRA 到基座（merge_and_unload）…")
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
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
    print(quick_generate_no_system(model, tokenizer, "早晨如何与主人互动？"))
    print(f"\n✅ 完成！LoRA 保存在: {args.output_dir}" + (f"；合并模型在: {args.merged_dir}" if args.do_merge else ""))

if __name__ == "__main__":
    main()
