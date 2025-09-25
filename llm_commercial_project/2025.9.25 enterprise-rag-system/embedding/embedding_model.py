import torch
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from config import config
from utils.logging import logger

class EmbeddingModel:
    """文本嵌入模型，用于将文本转换为向量表示"""
    
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        """
        初始化嵌入模型
        
        Args:
            model_name: 预训练模型名称
        """
        self.model_name = model_name
        self.device = config.DEVICE
        logger.info(f"加载嵌入模型: {model_name}，设备: {self.device}")
        
        # 加载模型
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder=str(config.CACHE_DIR / "sentence_transformers")
        )
        
        # 检查模型输出维度是否与配置一致
        self.dim = self.model.get_sentence_embedding_dimension()
        if self.dim != config.EMBEDDING_DIM:
            logger.warning(
                f"模型输出维度 {self.dim} 与配置的 {config.EMBEDDING_DIM} 不一致，"
                f"将使用模型实际维度 {self.dim}"
            )
            # 更新配置
            config.EMBEDDING_DIM = self.dim
        
        logger.info(f"嵌入模型加载完成，输出维度: {self.dim}")
    
    def encode(self, texts: List[str], batch_size: int = config.BATCH_SIZE) -> List[List[float]]:
        """
        将文本列表转换为向量嵌入
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            向量嵌入列表
        """
        if not texts:
            return []
            
        # 确保输入是列表
        if isinstance(texts, str):
            texts = [texts]
        
        # 使用模型编码文本
        logger.debug(f"编码文本，数量: {len(texts)}, 批大小: {batch_size}")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True  # 转换为numpy数组以便后续处理
        )
        
        # 转换为列表格式
        return embeddings.tolist()
    
    def encode_chunks(self, chunks: List[Dict]) -> List[Tuple[Dict, List[float]]]:
        """
        为分块列表生成嵌入
        
        Args:
            chunks: 分块列表，每个分块是包含"content"和"metadata"的字典
            
        Returns:
            分块及其对应嵌入的列表
        """
        if not chunks:
            return []
            
        # 提取文本内容
        texts = [chunk["content"] for chunk in chunks]
        
        # 生成嵌入
        embeddings = self.encode(texts)
        
        # 配对分块和嵌入
        return list(zip(chunks, embeddings))

# 创建嵌入模型实例
embedding_model = EmbeddingModel()
