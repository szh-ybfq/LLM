import torch
from pathlib import Path

class Config:
    """系统配置类，集中管理所有可配置参数"""
    
    # 基础配置
    PROJECT_NAME = "Enterprise RAG System"
    VERSION = "1.0.0"
    DEBUG = False
    
    # 路径配置
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    CACHE_DIR = BASE_DIR / "cache"
    
    # 创建必要的目录
    for dir_path in [DATA_DIR, LOGS_DIR, CACHE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 设备配置 - 优先使用CUDA
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        # 确保CUDA版本兼容性
        CUDA_VERSION = torch.version.cuda
        assert CUDA_VERSION.startswith("12.8"), f"需要CUDA 12.8，当前版本: {CUDA_VERSION}"
    
    # 嵌入模型配置
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # 高性能嵌入模型
    EMBEDDING_DIM = 1024  # 对应模型的输出维度
    BATCH_SIZE = 32  # 嵌入生成的批处理大小
    
    # 文档处理配置
    CHUNK_SIZE = 500  # 文档分块大小（字符数）
    CHUNK_OVERLAP = 50  # 块之间的重叠字符数
    MIN_CHUNK_SIZE = 100  # 最小块大小，过滤过短的块
    
    # 向量数据库配置
    MILVUS_HOST = "localhost"
    MILVUS_PORT = 19530
    MILVUS_COLLECTION = "enterprise_rag_collection"
    MILVUS_INDEX_TYPE = "HNSW"  # 高效的近似最近邻索引
    MILVUS_METRIC_TYPE = "L2"  # 距离度量方式
    
    # 检索配置
    VECTOR_SEARCH_TOP_K = 20  # 向量检索返回的候选数
    BM25_TOP_K = 20  # BM25检索返回的候选数
    HYBRID_TOP_K = 15  # 混合检索后的候选数
    RERANK_TOP_K = 5  # 重排序后的最终候选数
    
    # 重排序模型配置
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # LLM配置
    LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # 可替换为其他模型
    LLM_MAX_NEW_TOKENS = 512  # 生成回答的最大长度
    LLM_TEMPERATURE = 0.1  # 控制生成的随机性，越低越确定
    LLM_DEVICE = DEVICE
    
    # API配置
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_WORKERS = 4  # 工作进程数

# 创建配置实例
config = Config()
