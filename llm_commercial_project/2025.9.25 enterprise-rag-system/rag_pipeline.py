from typing import List, Dict, Tuple, Optional
from config import config
from utils.logging import logger
from data_processing.document_loader import document_loader
from data_processing.text_splitter import text_splitter
from embedding.embedding_model import embedding_model
from vector_db.milvus_client import milvus_client
from retrieval.advanced_retriever import advanced_retriever
from generation.llm_generator import llm_generator

class RAGPipeline:
    """完整的RAG流水线，整合所有组件"""
    
    def __init__(self):
        """初始化RAG流水线"""
        self.document_loader = document_loader
        self.text_splitter = text_splitter
        self.embedding_model = embedding_model
        self.vector_db = milvus_client
        self.retriever = advanced_retriever
        self.generator = llm_generator
        
        logger.info("RAG流水线初始化完成")
    
    def ingest_document(self, file_path: str) -> Tuple[int, int]:
        """
        处理并摄入单个文档到系统中
        
        Args:
            file_path: 文档路径
            
        Returns:
            元组 (总块数, 成功插入的块数)
        """
        try:
            # 1. 加载文档
            content, metadata = self.document_loader.load_document(file_path)
            
            # 2. 文本分块
            chunks = self.text_splitter.split_text(content, metadata)
            total_chunks = len(chunks)
            
            if total_chunks == 0:
                logger.warning(f"文档 {file_path} 分块后为空，跳过摄入")
                return (0, 0)
            
            logger.info(f"文档 {file_path} 分块完成，共 {total_chunks} 块")
            
            # 3. 生成嵌入
            chunk_embeddings = self.embedding_model.encode_chunks(chunks)
            
            # 4. 准备插入向量数据库的数据
            data_to_insert = []
            for chunk, embedding in chunk_embeddings:
                data_to_insert.append((chunk["content"], chunk["metadata"], embedding))
            
            # 5. 插入向量数据库
            inserted_ids = self.vector_db.insert(data_to_insert)
            inserted_count = len(inserted_ids)
            
            # 6. 添加到BM25语料库
            self.retriever.add_to_bm25(chunks)
            
            logger.info(f"文档 {file_path} 摄入完成，插入 {inserted_count}/{total_chunks} 块")
            return (total_chunks, inserted_count)
            
        except Exception as e:
            logger.error(f"文档 {file_path} 摄入失败: {str(e)}", exc_info=True)
            raise
    
    def ingest_directory(self, dir_path: str) -> Dict:
        """
        处理并摄入目录中的所有文档
        
        Args:
            dir_path: 目录路径
            
        Returns:
            统计信息字典
        """
        stats = {
            "total_documents": 0,
            "successful_documents": 0,
            "total_chunks": 0,
            "inserted_chunks": 0,
            "failed_documents": []
        }
        
        # 加载目录中的所有文档
        documents = self.document_loader.load_directory(dir_path)
        stats["total_documents"] = len(documents)
        
        logger.info(f"开始摄入目录 {dir_path} 中的 {len(documents)} 个文档")
        
        # 逐个处理文档
        for i, (content, metadata) in enumerate(documents, 1):
            file_name = metadata.get("file_name", f"未知文档_{i}")
            logger.info(f"处理文档 {i}/{len(documents)}: {file_name}")
            
            try:
                # 文本分块
                chunks = self.text_splitter.split_text(content, metadata)
                total_chunks = len(chunks)
                stats["total_chunks"] += total_chunks
                
                if total_chunks == 0:
                    logger.warning(f"文档 {file_name} 分块后为空，跳过摄入")
                    continue
                
                # 生成嵌入
                chunk_embeddings = self.embedding_model.encode_chunks(chunks)
                
                # 准备插入向量数据库的数据
                data_to_insert = []
                for chunk, embedding in chunk_embeddings:
                    data_to_insert.append((chunk["content"], chunk["metadata"], embedding))
                
                # 插入向量数据库
                inserted_ids = self.vector_db.insert(data_to_insert)
                inserted_count = len(inserted_ids)
                stats["inserted_chunks"] += inserted_count
                
                # 添加到BM25语料库
                self.retriever.add_to_bm25(chunks)
                
                stats["successful_documents"] += 1
                logger.info(f"文档 {file_name} 摄入完成，插入 {inserted_count}/{total_chunks} 块")
                
            except Exception as e:
                logger.error(f"文档 {file_name} 摄入失败: {str(e)}", exc_info=True)
                stats["failed_documents"].append({
                    "file_name": file_name,
                    "error": str(e)
                })
        
        logger.info(f"目录 {dir_path} 摄入完成，统计: {stats}")
        return stats
    
    def query(self, query: str) -> Dict:
        """
        处理用户查询，返回生成的回答
        
        Args:
            query: 用户查询文本
            
        Returns:
            包含回答和相关信息的字典
        """
        if not query or not query.strip():
            return {
                "query": query,
                "answer": "请提供有效的查询内容。",
                "contexts": [],
                "generated_tokens": 0
            }
        
        try:
            # 1. 检索相关上下文
            contexts = self.retriever.retrieve(query.strip())
            
            # 2. 生成回答
            result = self.generator.generate(query.strip(), contexts)
            
            # 3. 整理结果
            response = {
                "query": query,
                "answer": result["answer"],
                "contexts": result["contexts_used"],
                "generated_tokens": result["generated_tokens"],
                "retrieved_contexts_count": len(contexts)
            }
            
            logger.info(f"查询处理完成，生成 {result['generated_tokens']} 个token")
            return response
            
        except Exception as e:
            logger.error(f"查询处理失败: {str(e)}", exc_info=True)
            return {
                "query": query,
                "answer": f"处理查询时发生错误: {str(e)}",
                "contexts": [],
                "generated_tokens": 0,
                "retrieved_contexts_count": 0
            }
    
    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        return {
            "document_count": len(self.retriever.corpus_metadata),  # 近似值
            "chunk_count": self.vector_db.count(),
            "embedding_model": self.embedding_model.model_name,
            "llm_model": self.generator.model_name,
            "vector_db": {
                "name": "Milvus",
                "collection": self.vector_db.collection_name,
                "host": self.vector_db.host,
                "port": self.vector_db.port
            },
            "device": config.DEVICE
        }
    
    def clear_data(self) -> None:
        """清除所有数据（谨慎使用）"""
        # 删除向量数据库中的所有数据
        self.vector_db.drop_collection()
        # 重新初始化集合
        self.vector_db._ensure_collection()
        # 清空BM25语料库
        self.retriever.bm25_corpus = []
        self.retriever.corpus_metadata = []
        self.retriever.bm25 = None
        
        logger.warning("系统中的所有数据已清除")

# 创建RAG流水线实例
rag_pipeline = RAGPipeline()
