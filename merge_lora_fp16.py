#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_gemma3n_neko_lora.py
将 LoRA 适配器合并回基座模型（保存为 safetensors）。
路径硬编码，直接运行即可。
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ===== 硬编码路径 / 模型名 =====
BASE_MODEL = "unsloth/gemma-3n-E4B-it"   # 基座模型
LORA_DIR   = "gemma3n-neko-lora"         # 训练输出的 LoRA 适配器目录
MERGED_DIR = "gemma3n-neko-lora-merged-fp16"  # 合并后输出目录

def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32

def main():
    # 基础检查
    if not os.path.isdir(LORA_DIR):
        print(f"[ERROR] 未找到 LoRA 目录：{LORA_DIR}")
        sys.exit(1)

    dtype = pick_dtype()
    device_map = "auto" if torch.cuda.is_available() else None

    print(f"[1/4] 加载基座模型: {BASE_MODEL} (dtype={dtype}, device_map={device_map})")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"[2/4] 加载 LoRA 适配器: {LORA_DIR}")
    peft = PeftModel.from_pretrained(base, LORA_DIR)

    print("[3/4] 合并权重（merge_and_unload）…")
    merged = peft.merge_and_unload()

    os.makedirs(MERGED_DIR, exist_ok=True)
    print(f"[4/4] 保存合并模型到: {MERGED_DIR}")
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)

    # 保存 tokenizer（优先使用 LoRA 目录，保留 chat_template / pad_token 等自定义）
    tok_src_lora = os.path.join(LORA_DIR, "tokenizer_config.json")
    tok_src = LORA_DIR if os.path.exists(tok_src_lora) else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_DIR)

    print("✅ 合并完成！")
    print(f"   - LoRA:   {os.path.abspath(LORA_DIR)}")
    print(f"   - 基座:   {BASE_MODEL}")
    print(f"   - 输出:   {os.path.abspath(MERGED_DIR)}")
    print("提示：若显存不足，可把 device_map=None 改为 CPU，或降低 dtype 到 float32。")

if __name__ == "__main__":
    main()
