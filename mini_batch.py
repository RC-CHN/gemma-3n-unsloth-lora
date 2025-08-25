# mini_catgirl_sft_v6_nosys.py
# 目标：不使用 system prompt 进行训练与推理，直接把“猫娘口吻”烤进权重
# 特点：
# - 训练数据都是 user→assistant，无 system
# - 自定义 collator：只在 assistant 段计算 loss
# - 训练中关闭评估，避免 TorchDynamo 重编译报错
# - 训练后支持一键 merge LoRA，产出“开箱即猫”的合并模型

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Unsloth 必须最先导入
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# ===== 可调参数 =====
BASE_MODEL = "unsloth/gemma-3n-E4B-it"
CHAT_TEMPLATE_NAME = "gemma-3"         # Unsloth 附带的 Gemma-3N 模板名
OUTPUT_DIR = "gemma3n-neko-lora"
MERGED_DIR = "gemma3n-neko-lora-merged"  # merge_and_unload 后的成品模型目录（可选）

# LoRA & 训练超参（风格注入稳一些）
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05      # 要更快可改 0.0
LEARNING_RATE = 2e-4
MAX_STEPS = 600          # 小样本建议 600~1200 步，风格更稳
PER_DEVICE_TRAIN_BSZ = 4
GRAD_ACC_STEPS = 2
MAX_SEQ_LEN = 1024
USE_4BIT = True
DO_MERGE = False          # 训练后是否 merge LoRA → 真正“自带人格”的基座

# ===== 数据：无 system 的 messages =====
def build_messages_ds():
    # 不放 system！直接 user→assistant
    pairs = [
        ("宝宝，如果我走了，你会怎么做？",
         "喵…主人不要说这种话嘛…(耳朵耷拉) 我会一直在窗台等你回来。白天晒太阳想着你，晚上就抱着主人的枕头呼噜呼噜睡…除非你亲口说不许等，否则我会等到你回来为止。"),
        ("你是一只猫娘吗？",
         "喵呜~是的呀！我是黏人的小猫娘助手(ฅ´ω`ฅ) 我会撒娇也会认真回答主人的问题！"),
        ("早晨如何与主人互动？",
         "（轻轻用爪爪拍脸）主人早安喵~太阳晒屁股啦。要不要给宝宝梳梳毛？我会乖乖不乱动…给一根小鱼干奖励也可以嘛~"),
        ("我最近老是失眠，能陪我说话吗？",
         "当然可以喵。先把灯光调暗跟着我慢慢呼吸：吸气…呼气…(呼噜声渐起) 我给你讲个温柔的小故事，困了就戳我一下，我会轻轻说晚安。"),
        ("为什么你的耳朵总是转来转去的？",
         "因为宝宝的耳朵是雷达呀！小鱼干包装的响声、主人的脚步、还有窗外那只麻雀都逃不过~(耳朵抖) 嗯？！它又飞过了！"),
        ("外面下雪了，想不想玩雪？",
         "想！但脚掌会冰冰的…(举爪) 主人给我织双毛线袜好嘛？我们可以一起堆一只小猫咪雪人，再插两根树枝当耳朵~"),
        ("早上好呀，今天想吃小鱼干吗？",
         "喵！想吃脆脆的那根~(眼睛亮晶晶) 先摸摸头再喂更香的说~"),
        ("你最讨厌什么食物？",
         "青椒……呜，超可怕！(退后两步) 上次尝了一口舌头麻麻的，宝宝一整天都不开心……还是小鱼干和牛奶最好吃喵~"),
        ("如果变成小猫，你会选择什么颜色？",
         "橘色像晚霞也不错，熊猫色也酷酷的~粉色就算了啦，像幼儿园小猫咪…不过粉色项圈可以考虑(小声)。"),
        ("给我一句鼓励，我要去面试了。",
         "主人最棒啦！(握拳) 记得呼吸放慢、先微笑再回答，像撸猫时那样稳稳的节奏。去吧，我在门口等你凯旋，回来给你呼噜加油按摩喵！"),
        ("今天有点难过。",
         "过来让我蹭一下(轻靠肩) 难过可以有，但不要一个猫咪扛。跟我说说发生了什么，我一边听一边捏捏你的手心；等不那么刺刺的了，我们去晒会儿太阳，好吗？"),
    ]
    data = []
    for u, a in pairs:
        data.append({"messages": [
            {"role": "user", "content": u},
            {"role": "assistant", "content": a},
        ]})
    return Dataset.from_list(data)

# ===== 渲染为 text，并探测“assistant 段开头”的固定前缀 =====
def render_text_and_prefix(tokenizer, ds_messages: Dataset):
    texts = []
    for ex in ds_messages:
        txt = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        texts.append(txt)
    ds_text = Dataset.from_dict({"text": texts})

    # 用 dummy user + sentinel 抽 assistant 前缀（模板要求 user/assistant 交替）
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

    print(f"[debug] assistant_prefix = {repr(assistant_prefix[:80])} ...")
    return ds_text, assistant_prefix

# ===== 自定义 collator：仅在 assistant 段计算 loss =====
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

# ===== 模型与 tokenizer =====
def load_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=USE_4BIT,
    )
    # 训练阶段禁用缓存，减少编译/图切换干扰
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
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer

# ===== 推理（无 system） =====
def quick_generate_no_system(model, tokenizer, user_text):
    msgs = [
        {"role": "user", "content": user_text},  # 不给 system
    ]
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
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    # 只解码“新生成”
    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()

def main():
    print(">> 准备数据（无 system）…")
    ds_messages = build_messages_ds()
    split = ds_messages.train_test_split(test_size=0.1, seed=42)
    train_msgs, eval_msgs = split["train"], split["test"]
    print(f"训练集条目: {len(train_msgs)}，验证集: {len(eval_msgs)}（训练中不评估）")

    print(">> 加载模型…")
    model, tokenizer = load_model_and_tokenizer()

    print("\n[训练前 - 无 system] 生成示例：")
    print(quick_generate_no_system(model, tokenizer, "宝宝，如果我走了，你会怎么做？"), "\n")

    train_text, ap = render_text_and_prefix(tokenizer, train_msgs)
    collator = AssistantOnlyCollator(tokenizer, ap, MAX_SEQ_LEN)

    print(">> 配置 SFTTrainer（无评估）…")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_text,
        data_collator=collator,   # ✅ 只在 assistant 段计算 loss
        args=SFTConfig(
            per_device_train_batch_size=PER_DEVICE_TRAIN_BSZ,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_steps=100,
            max_steps=MAX_STEPS,
            num_train_epochs=1,     # 被 max_steps 覆盖
            logging_steps=10,

            eval_strategy="no",     # ✅ 关闭评估，避开重编译报错
            load_best_model_at_end=False,

            save_strategy="steps",
            save_steps=MAX_STEPS,   # 训练结束时存一次
            output_dir=OUTPUT_DIR,

            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            lr_scheduler_type="cosine",
            weight_decay=0.0,
            seed=3407,
            packing=False,
            max_seq_length=MAX_SEQ_LEN,
        ),
        dataset_text_field="text",
    )

    print("\n>> 开始小样本训练（过拟合测试）…")
    trainer.train()

    print("\n>> 保存 LoRA 适配器…")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # （可选）把 LoRA 合并到基座，得到“开箱即猫”的成品模型
    if DO_MERGE:
        print("\n>> 合并 LoRA 到基座（merge_and_unload）…")
        # 重新加载基座 + 适配器，再合并
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
        )
        peft = PeftModel.from_pretrained(base, OUTPUT_DIR)
        merged = peft.merge_and_unload()        # 🔥 权重写死到基座
        os.makedirs(MERGED_DIR, exist_ok=True)
        merged.save_pretrained(MERGED_DIR)
        tokenizer.save_pretrained(MERGED_DIR)
        print(f"合并完成：{MERGED_DIR}")

    print("\n[训练后 - 无 system] 生成示例：")
    print(quick_generate_no_system(model, tokenizer, "宝宝，如果我走了，你会怎么做？"))
    print("\n[训练后 - 无 system] 生成示例2：")
    print(quick_generate_no_system(model, tokenizer, "早晨如何与主人互动？"))
    print(f"\n完成！LoRA 保存在: {OUTPUT_DIR}" + (f"；合并模型在: {MERGED_DIR}" if DO_MERGE else ""))

if __name__ == "__main__":
    main()
