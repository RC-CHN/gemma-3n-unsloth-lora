import json
import argparse
import os

def convert_to_sharegpt(input_file, output_file):
    """
    将包含 "instruction" 和 "output" 的 JSON 文件转换为 ShareGPT 格式的 JSONL 文件。

    Args:
        input_file (str): 输入的 JSON 文件路径。
        output_file (str): 输出的 JSONL 文件路径。
    """
    print(f"开始转换文件: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误: 输入文件 '{input_file}' 不是有效的 JSON 格式。")
        return
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file}' 未找到。")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    converted_count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in data:
            if "instruction" in item and "output" in item:
                conversation = {
                    "conversations": [
                        {"from": "human", "value": item["instruction"]},
                        {"from": "gpt", "value": item["output"]}
                    ]
                }
                f_out.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                converted_count += 1
    
    print(f"转换完成！总共转换了 {converted_count} 条记录。")
    print(f"转换后的文件已保存至: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="将 instruction/output 格式的 JSON 转换为 ShareGPT 格式的 JSONL。")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入的 JSON 文件路径 (例如: dataset/NekoQA-10K.json)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出的 JSONL 文件路径 (例如: dataset/NekoQA-10K_converted.jsonl)"
    )
    args = parser.parse_args()
    
    convert_to_sharegpt(args.input_file, args.output_file)

if __name__ == "__main__":
    main()