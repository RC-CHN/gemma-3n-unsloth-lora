#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 [{"instruction": "...", "output": "..."}] 转成 messages JSONL：
{"messages":[{"role":"user","content":...},{"role":"assistant","content":...}]}

用法：
  python convert_inst2messages.py \
    --input data.json \          # 支持 .json（数组）或 .jsonl
    --output catgirl_messages.jsonl \
    --with-system \              # 可选：加一段固定人设
    --persona "你是可爱黏人的猫娘助手，称呼用户为“主人”。说话自然、有拟声词，且有信息量。"

转换后可直接被你的 SFT 脚本读取（load_dataset('json', data_files=..., split='train')）。
"""
import argparse, json, os, sys, io, random
from typing import Iterable, Dict

def read_json_or_jsonl(path: str) -> Iterable[Dict]:
    if path.endswith(".jsonl"):
        with io.open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    yield obj
                except Exception as e:
                    sys.stderr.write(f"[warn] 跳过一行（JSON 解析失败）：{e}\n")
    else:
        with io.open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # 兼容 {"data":[...]}
            data = data.get("data", [])
        if not isinstance(data, list):
            raise ValueError("JSON 顶层应为数组或含 data 数组的对象。")
        for obj in data:
            yield obj

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return s.strip()

def convert_item(obj: Dict, add_system: bool, persona: str) -> Dict:
    instr = normalize_text(obj.get("instruction", ""))
    out   = normalize_text(obj.get("output", ""))
    if not instr or not out:
        raise ValueError("缺少 instruction 或 output")
    msgs = []
    if add_system and persona:
        msgs.append({"role": "system", "content": persona})
    msgs.append({"role": "user", "content": instr})
    msgs.append({"role": "assistant", "content": out})
    return {"messages": msgs}

def main():
    ap = argparse.ArgumentParser(description="Convert instruction/output to messages JSONL")
    ap.add_argument("--input", required=True, help="输入文件：.json（数组）或 .jsonl")
    ap.add_argument("--output", required=True, help="输出 JSONL 路径")
    ap.add_argument("--with-system", action="store_true", help="是否为每条样本添加固定 system 人设")
    ap.add_argument("--persona", type=str, default="你是可爱黏人的猫娘助手，称呼用户为“主人”。说话自然、轻松、带拟声词和表情，但回答要有信息量，遵守安全与法律。",
                    help="--with-system 时使用的人设文本")
    ap.add_argument("--shuffle", action="store_true", help="输出前随机打乱样本")
    ap.add_argument("--seed", type=int, default=42, help="shuffle 随机种子")
    args = ap.parse_args()

    items = []
    for obj in read_json_or_jsonl(args.input):
        try:
            items.append(convert_item(obj, args.with_system, args.persona))
        except Exception as e:
            sys.stderr.write(f"[warn] 跳过一条样本：{e}\n")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(items)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with io.open(args.output, "w", encoding="utf-8") as w:
        for ex in items:
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ 转换完成：{len(items)} 条")
    # 打印一条样例预览
    if items:
        preview = json.dumps(items[0], ensure_ascii=False, indent=2)
        print("— 示例 —\n" + preview)

if __name__ == "__main__":
    main()
