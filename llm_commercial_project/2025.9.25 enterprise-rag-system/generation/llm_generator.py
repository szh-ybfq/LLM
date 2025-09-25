import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import config
from utils.logging import logger

class LLMGenerator:
    """LLM生成器，基于检索到的上下文生成回答"""
    
    def __init__(self, model_name: str = config.LLM_MODEL):
        """
        初始化LLM生成器
        
        Args:
            model_name: 预训练模型名称
        """
        self.model_name = model_name
        self.device = config.LLM_DEVICE
        
        # 配置量化参数（4位量化以节省内存）
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(config.CACHE_DIR / "transformers"),
            use_fast=True
        )
        
        # 设置pad_token（如果模型没有）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.quantization_config,
            device_map="auto",
            cache_dir=str(config.CACHE_DIR / "transformers"),
            trust_remote_code=True
        )
        
        # 设置为评估模式
        self.model.eval()
        
        logger.info(f"LLM生成器初始化完成，模型: {model_name}，设备: {self.device}")
    
    def _build_prompt(self, query: str, contexts: List[Dict]) -> str:
        """
        构建提示词
        
        Args:
            query: 用户查询
            contexts: 检索到的上下文列表
            
        Returns:
            构建好的提示词
        """
        # 系统提示
        system_prompt = (
            "你是一个基于检索信息回答问题的助手。请严格根据提供的上下文信息回答问题，"
            "不要编造信息。如果上下文信息不足以回答问题，请明确说明。"
            "回答要简洁明了，准确无误，并在回答末尾标注信息来源。"
        )
        
        # 构建上下文部分
        context_str = ""
        for i, context in enumerate(contexts, 1):
            content = context["content"]
            source = context["metadata"].get("file_name", "未知来源")
            context_str += f"上下文 {i}（来源：{source}）:\n{content}\n\n"
        
        # 构建用户查询部分
        user_prompt = f"问题: {query}\n\n请根据以上上下文回答问题:"
        
        # 组合完整提示（使用Llama风格的格式）
        prompt = (
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            f"{context_str}{user_prompt} [/INST]"
        )
        
        return prompt
    
    def generate(
        self,
        query: str,
        contexts: List[Dict],
        max_new_tokens: int = config.LLM_MAX_NEW_TOKENS,
        temperature: float = config.LLM_TEMPERATURE
    ) -> Dict:
        """
        基于检索到的上下文生成回答
        
        Args:
            query: 用户查询
            contexts: 检索到的上下文列表
            max_new_tokens: 生成的最大token数
            temperature: 生成温度，控制随机性
            
        Returns:
            包含生成回答和相关信息的字典
        """
        if not contexts:
            return {
                "answer": "抱歉，没有找到相关信息来回答这个问题。",
                "contexts_used": [],
                "generated_tokens": 0
            }
        
        # 构建提示词
        prompt = self._build_prompt(query, contexts)
        logger.debug(f"生成提示词: {prompt}")
        
        # 分词
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # 根据模型最大上下文调整
        ).to(self.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1  # 减少重复
            )
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],  # 只取新生成的部分
            skip_special_tokens=True
        )
        
        # 统计生成的token数
        generated_tokens = len(outputs[0]) - inputs.input_ids.shape[1]
        
        logger.info(f"生成完成，生成token数: {generated_tokens}")
        
        return {
            "answer": generated_text,
            "contexts_used": contexts,
            "generated_tokens": generated_tokens
        }

# 创建LLM生成器实例
llm_generator = LLMGenerator()
