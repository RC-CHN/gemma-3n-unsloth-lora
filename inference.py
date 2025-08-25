from unsloth import FastLanguageModel
import torch

# --- 模型配置 ---
# 基础模型名称 (与训练时使用的模型一致)
base_model_name = "unsloth/gemma-3n-E4B-it"
# LoRA适配器路径 (训练脚本中的 output_dir)
lora_adapter_path = "gemma-3n-finetuned"

# --- 加载参数 ---
# 如果你的GPU支持 bfloat16 (如 Ampere, Hopper架构)，设为 torch.bfloat16 以获得更好性能
# 对于旧款GPU (T4, V100)，Float16 会被自动使用
dtype = None 
# 与训练脚本保持一致
load_in_4bit = True

print("正在从LoRA适配器加载模型...")
# Unsloth可以非常方便地直接从LoRA适配器加载模型
# 它会自动处理基础模型的下载和与适配器的合并
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=lora_adapter_path,
    max_seq_length=2048,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

print("模型加载完毕。")

# --- 推理 ---
# 我们将使用Unsloth内置的聊天模板功能
# 这能确保输入格式与模型微调时完全一致
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3", # 使用与训练时相同的模板
)

# 聊天记录
messages = []

print("\n--- 开始交互式聊天 ---")
print("输入 'exit' 或 'quit' 退出。")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # Gemma-3N 期望一个内容部分的列表，即使只有文本
    messages.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
    
    # 将聊天记录格式化为模型输入
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    # 从输入中获取当前tokens的数量，以便之后只解码新生成的部分
    input_length = inputs.shape[1]
    
    # 生成回复
    outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True)
    
    # 只解码新生成的部分，跳过输入的tokens
    new_tokens = outputs[0, input_length:]
    model_response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"Model: {model_response}")
    
    # 将模型的回复也加入聊天记录
    messages.append({"role": "assistant", "content": [{"type": "text", "text": model_response}]})

print("--- 聊天结束 ---")