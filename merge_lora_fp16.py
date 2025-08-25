#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_lora_fp16.py
把 LoRA 适配器合并到基座，导出单一权重目录（默认 BF16，更稳；也可选 FP16）。
"""

import argparse, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from unsloth.chat_templates import get_chat_template  # 仅用于恢复 chat_template

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="unsloth/gemma-3n-E4B-it", help="基座模型 id/路径")
    ap.add_argument("--lora_dir",   default="gemma3n-neko-lora", help="训练好的 LoRA 目录")
    ap.add_argument("--out_dir",    default="gemma3n-neko-merged-fp16", help="合并输出目录")
    ap.add_argument("--dtype", choices=["bf16","fp16"], default="bf16", help="保存时用的权重精度")
    ap.add_argument("--chat_template", default="gemma-3", help="要写入 tokenizer 的 chat 模板名")
    args = ap.parse_args()

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    os.makedirs(args.out_dir, exist_ok=True)

    print(f">> 加载基座: {args.base_model}  ({args.dtype})")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch_dtype, device_map="auto"
    )
    print(f">> 加载 LoRA: {args.lora_dir} 并合并 …")
    peft = PeftModel.from_pretrained(base, args.lora_dir)
    merged = peft.merge_and_unload()   # 🔥 权重写死到基座

    print(f">> 保存合并模型到: {args.out_dir}")
    merged.save_pretrained(args.out_dir, safe_serialization=True)

    # tokenizer：优先用 LoRA 目录（含你的新增特殊符号/模板），否则回退到基座再写入 chat 模板
    try:
        tok = AutoTokenizer.from_pretrained(args.lora_dir, use_fast=True)
        print(">> 使用 LoRA 目录中的 tokenizer")
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        print(">> 使用基座 tokenizer 并写入 chat_template")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok = get_chat_template(tok, chat_template=args.chat_template)
    tok.save_pretrained(args.out_dir)

    print("✅ 合并完成！")

if __name__ == "__main__":
    main()
