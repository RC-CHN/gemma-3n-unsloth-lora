import argparse
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def main():
    parser = argparse.ArgumentParser(description="使用Unsloth微调Gemma-3N模型")
    
    # 模型与分词器参数
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3n-E4B-it", help="要使用的基础模型名称")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--load_in_4bit", action='store_true', default=True, help="是否以4-bit加载模型")
    
    # 数据集参数
    parser.add_argument("--dataset_path", type=str, required=True, help="转换后的.jsonl数据集文件的路径")
    parser.add_argument("--chat_template", type=str, default="gemma-3", help="要应用的聊天模板")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="gemma-3n-finetuned", help="保存微调后适配器的目录")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="训练的总轮数")
    parser.add_argument("--max_steps", type=int, default=-1, help="最大训练步数（覆盖num_train_epochs）")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个设备的训练批量大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--logging_steps", type=int, default=1, help="记录日志的步数间隔")
    
    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA的秩")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA的alpha值")

    args = parser.parse_args()

    # 1. 加载模型和分词器
    print(f"正在加载模型: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    # 2. 添加LoRA适配器
    print("正在配置LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 3. 准备数据集
    print(f"正在加载数据集: {args.dataset_path}")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=args.chat_template,
    )
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>') for convo in convos]
        return {"text": texts}

    dataset = load_dataset("json", data_files={"train": args.dataset_path}, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 4. 配置训练器
    print("正在配置训练器...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False, 
        args=SFTConfig(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
            report_to="none",
        ),
    )

    # 5. 开始训练
    print("开始模型训练...")
    trainer_stats = trainer.train()

    print("\n训练完成!")
    print(f"训练耗时: {trainer_stats.metrics['train_runtime']} 秒")
    
    # 6. 保存模型
    print(f"正在将模型适配器保存到: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("模型保存完毕。")

if __name__ == "__main__":
    main()