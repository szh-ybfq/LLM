# %%
import os
import torch
import deepspeed
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    set_seed
)
from transformers.integrations import HfDeepSpeedConfig

# %%
import mpi4py

# %%
import pyarrow
print("当前 pyarrow 版本：", pyarrow.__version__)
# 检查是否存在 PyExtensionType（正常应输出 <class 'pyarrow._ext.PyExtensionType'>）
print("是否有 PyExtensionType：", hasattr(pyarrow, "PyExtensionType"))

# %%
# 执行以下代码（复制粘贴）
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

print("=== 环境验证 ===")
print("Python 路径：", transformers.__file__.split("/lib/")[0])  # 应包含 "qwen2_env"
print("transformers 版本：", transformers.__version__)  # 必须是 4.38.0
print("是否支持 'qwen2'：", "qwen2" in CONFIG_MAPPING)  # 必须输出 True
print("Qwen2Config 是否存在：", hasattr(transformers.models.qwen2, "Qwen2Config"))  # 必须是 True

# %% [markdown]
# # ==============================================
# # 1. 配置参数解析模块：解析命令行参数和DeepSpeed配置
# # ==============================================

# %%

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-R1-Distill-Qwen-14B 训练脚本（DeepSpeed加速）")
    # 基础训练参数
    parser.add_argument("--model_name_or_path", type=str, default="./models/DeepSeek-R1-Distill-Qwen-14B",  help="预训练模型路径")
    parser.add_argument("--dataset_name", type=str, default="togethercomputer/RedPajama-Data-1T",  help="大规模数据集名称（RedPajama-1T约1TB文本数据）")
    parser.add_argument("--dataset_split", type=str, default="train", help="数据集拆分（train/validation）")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="模型保存路径")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="单GPU训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # DeepSpeed参数（通过配置文件指定）
    parser.add_argument("--deepspeed", type=str, default="ds_config.json" , help="DeepSpeed配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练本地进程ID（DeepSpeed自动设置）")
    
    # 关键修改：用parse_known_args()，忽略Jupyter的--f等未知参数
    args, _ = parser.parse_known_args()  # 返回 (已知参数, 未知参数)，用_丢弃未知参数
    # 为什么这样改？
    # parse_known_args()会只解析你定义过的参数（如--model_name_or_path），自动忽略 Jupyter 传递的--f等未定义参数，彻底解决 “unrecognized arguments” 报错。
    return args


# %% [markdown]
# # ==============================================
# # 2. 数据预处理模块：处理大规模文本数据集
# # ==============================================

# %%

def prepare_dataset(tokenizer, args):
    """
    加载并预处理RedPajama-1T大规模数据集
    RedPajama-1T包含书籍、论文、网页等多源文本，适合语言模型预训练
    """
    # 加载数据集（仅加载必要列，减少内存占用）
    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        streaming=False,  # 非流式加载（需足够磁盘空间，约1TB）
        cache_dir="./dataset_cache"  # 缓存路径，避免重复下载
    )
    
    # 过滤空文本和过短文本
    def filter_func(example):
        return len(example["text"].strip()) > 100  # 保留长度>100的文本
    dataset = dataset.filter(filter_func, num_proc=os.cpu_count())  # 多进程过滤
    
    # 文本分词与截断
    def tokenize_func(example):
        # 分词，最大长度512（Qwen模型默认上下文长度）
        return tokenizer(
            example["text"],
            max_length=512,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=False  # 不处理超长文本的截断分片
        )
    
    # 多进程分词（加速处理大规模数据）
    tokenized_dataset = dataset.map(
        tokenize_func,
        batched=True,  # 批量处理
        num_proc=os.cpu_count(),
        remove_columns=["text"]  # 移除原始文本列，节省内存
    )
    
    # 格式化数据集为PyTorch张量
    tokenized_dataset = tokenized_dataset.with_format("torch", columns=["input_ids", "attention_mask"])
    
    # 构建标签（自回归任务：标签=输入ID）
    tokenized_dataset = tokenized_dataset.map(
        lambda x: {"labels": x["input_ids"].clone()},
        batched=True
    )
    
    return tokenized_dataset


# %% [markdown]
# # ==============================================
# # 3. 模型加载模块：加载预训练模型并配置量化（可选）
# # ==============================================

# %%

def load_model_and_tokenizer(args):
    """
    加载DeepSeek-R1-Distill-Qwen-14B模型和分词器
    支持4/8位量化以节省显存（14B模型全精度约需56GB显存）
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,  # Qwen系列需要信任远程代码
        padding_side="left"  # 自回归模型通常左padding
    )
    # 设置填充符（如未定义）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 配置量化参数（4位量化，适合显存有限的场景）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4位量化
        bnb_4bit_use_double_quant=True,  # 双量化优化
        bnb_4bit_quant_type="nf4",  # 归一化浮点4位
        bnb_4bit_compute_dtype=torch.float16  # 计算 dtype
    )
    
    # 加载模型（因果语言模型）
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        quantization_config=bnb_config,  # 应用量化
        # device_map="auto",  # 自动分配设备, zero3会自主管理
        torch_dtype=torch.float16  # 模型参数 dtype
    )
    
    # 禁用梯度检查点（如需节省显存可开启，但会降低速度）
    model.gradient_checkpointing_enable()
    
    return model, tokenizer



# %% [markdown]
# 
# # ==============================================
# # 4. DeepSpeed训练模块：配置并启动分布式训练
# # ==============================================
# 

# %%
def main():
    args = parse_args()
    set_seed(args.seed)  # 设置随机种子，保证可复现性
    # 初始化DeepSpeed配置（必须在模型加载前）
    dschf = HfDeepSpeedConfig(args.deepspeed) if args.deepspeed else None
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args)
    
    # 准备数据集
    tokenized_dataset = prepare_dataset(tokenizer, args)
    
    # 数据整理器（用于语言模型训练的批量处理）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 自回归模型不使用掩码语言模型（MLM）
    )
    
    # 配置DeepSpeed训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,  # 每10步记录一次日志
        save_steps=100,  # 每100步保存一次模型
        save_total_limit=3,  # 最多保存3个模型 checkpoint
        deepspeed=args.deepspeed,  # 启用DeepSpeed
        local_rank=args.local_rank,  # 分布式训练rank
        fp16=True,  # 启用混合精度训练
        report_to="tensorboard",  # 日志报告到TensorBoard
        remove_unused_columns=False,  # 保留所有列（避免标签被删除）
        gradient_accumulation_steps=4,  # 梯度累积（变相增大batch size）
        weight_decay=0.01,  # 权重衰减，防止过拟合
        warmup_steps=100,  # 学习率热身步数
    )
    
    # 初始化DeepSpeed训练器
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=args.deepspeed,
        args=training_args,
    )
    
    # ==============================================
    # 训练循环
    # ==============================================
    model.train()
    # 生成数据加载器（分布式场景下自动分片）
    train_loader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    
    # 迭代训练
    for epoch in range(args.num_train_epochs):
        print(f"===== Epoch {epoch + 1}/{args.num_train_epochs} =====")
        total_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            # 数据移动到设备
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch, use_cache=False)  # 禁用缓存以节省显存
            loss = outputs.loss
            
            # 反向传播（DeepSpeed自动处理梯度累积和同步）
            model.backward(loss)
            model.step()  # 优化器更新
            
            # 记录损失
            total_loss += loss.item()
            
            # 打印日志
            if (step + 1) % training_args.logging_steps == 0:
                avg_loss = total_loss / training_args.logging_steps
                print(f"Step {step + 1}, Average Loss: {avg_loss:.4f}")
                total_loss = 0.0
            
            # 保存模型
            if (step + 1) % training_args.save_steps == 0:
                model.save_checkpoint(f"{args.output_dir}/checkpoint-{epoch}-{step}")
        
        # 每个epoch结束保存一次模型
        model.save_checkpoint(f"{args.output_dir}/epoch-{epoch + 1}")
    
    print("训练完成！")



# %% [markdown]
# # 执行

# %%

if __name__ == "__main__":
    main()



