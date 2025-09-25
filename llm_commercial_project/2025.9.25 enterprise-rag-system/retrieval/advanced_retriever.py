import re
import torch
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import config
from embedding.embedding_model import embedding_model
from vector_db.milvus_client import milvus_client
from utils.logging import logger

class AdvancedRetriever:
    """高级检索器，结合向量检索、关键词检索和重排序"""
    
    def __init__(self):
        """初始化高级检索器"""
        # 初始化BM25所需的语料库（会在添加文档时更新）
        self.bm25_corpus = []
        self.bm25 = None
        self.corpus_metadata = []  # 存储语料库的元数据
        
        # 初始化重排序模型
        self.rerank_model_name = config.RERANK_MODEL
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(
            self.rerank_model_name,
            cache_dir=str(config.CACHE_DIR / "transformers")
        )
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            self.rerank_model_name,
            cache_dir=str(config.CACHE_DIR / "transformers")
        ).to(config.DEVICE)
        
        # 设置为评估模式
        self.rerank_model.eval()
        
        logger.info(f"高级检索器初始化完成，重排序模型: {self.rerank_model_name}")
    
    def add_to_bm25(self, chunks: List[Dict]) -> None:
        """
        将分块添加到BM25语料库
        
        Args:
            chunks: 分块列表
        """
        # 提取文本内容并进行预处理
        new_documents = []
        new_metadata = []
        
        for chunk in chunks:
            content = chunk["content"]
            # 文本预处理： lowercase, 去除特殊字符, 分词
            processed = self._preprocess_text(content)
            new_documents.append(processed)
            new_metadata.append(chunk["metadata"])
        
        # 更新语料库
        self.bm25_corpus.extend(new_documents)
        self.corpus_metadata.extend(new_metadata)
        
        # 重新初始化BM25
        self.bm25 = BM25Okapi(self.bm25_corpus)
        logger.info(f"BM25语料库更新，总文档数: {len(self.bm25_corpus)}")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理，用于BM25"""
        # 转为小写
        text = text.lower()
        # 去除特殊字符
        text = re.sub(r'[^\w\s]', ' ', text)
        # 分词
        tokens = text.split()
        return tokens
    
    def vector_search(self, query: str, limit: int = config.VECTOR_SEARCH_TOP_K) -> List[Dict]:
        """
        向量检索
        
        Args:
            query: 查询文本
            limit: 返回结果数量
            
        Returns:
            检索结果列表
        """
        # 生成查询向量
        query_embedding = embedding_model.encode([query])[0]
        
        # 执行向量检索
        results = milvus_client.search(
            query_embedding=query_embedding,
            limit=limit
        )
        
        # 添加检索类型标记
        for result in results:
            result["retrieval_type"] = "vector"
        
        return results
    
    def keyword_search(self, query: str, limit: int = config.BM25_TOP_K) -> List[Dict]:
        """
        关键词检索（BM25）
        
        Args:
            query: 查询文本
            limit: 返回结果数量
            
        Returns:
            检索结果列表
        """
        if not self.bm25 or not self.bm25_corpus:
            logger.warning("BM25语料库为空，无法执行关键词检索")
            return []
        
        # 预处理查询
        processed_query = self._preprocess_text(query)
        
        # 执行BM25检索
        scores = self.bm25.get_scores(processed_query)
        
        # 获取排名前limit的索引
        top_indices = scores.argsort()[-limit:][::-1]  # 从高到低排序
        
        # 构建结果
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只保留有分数的结果
                results.append({
                    "content": self.bm25_corpus[idx],
                    "metadata": self.corpus_metadata[idx],
                    "score": float(scores[idx]),
                    "retrieval_type": "keyword"
                })
        
        return results
    
    def hybrid_search(self, query: str) -> List[Dict]:
        """
        混合检索（向量检索 + 关键词检索）
        
        Args:
            query: 查询文本
            
        Returns:
            合并并去重后的检索结果
        """
        # 执行两种检索
        vector_results = self.vector_search(query)
        keyword_results = self.keyword_search(query)
        
        logger.debug(f"向量检索结果数: {len(vector_results)}, 关键词检索结果数: {len(keyword_results)}")
        
        # 合并结果并去重（基于内容）
        combined = {}
        for result in vector_results + keyword_results:
            # 使用内容的哈希作为唯一键，避免重复
            content_hash = hash(result["content"])
            if content_hash not in combined:
                combined[content_hash] = result
            else:
                # 如果内容相同，保留分数较高的结果
                if result["score"] > combined[content_hash]["score"]:
                    combined[content_hash] = result
        
        # 转换为列表并按分数排序
        combined_results = list(combined.values())
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        # 限制结果数量
        if len(combined_results) > config.HYBRID_TOP_K:
            combined_results = combined_results[:config.HYBRID_TOP_K]
        
        logger.debug(f"混合检索去重后结果数: {len(combined_results)}")
        return combined_results
    
    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        对候选结果进行重排序
        
        Args:
            query: 查询文本
            candidates: 候选结果列表
            
        Returns:
            重排序后的结果列表
        """
        if not candidates:
            return []
        
        # 准备输入
        pairs = [[query, candidate["content"]] for candidate in candidates]
        
        # 分词
        inputs = self.rerank_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(config.DEVICE)
        
        # 推理
        with torch.no_grad():
            outputs = self.rerank_model(**inputs)
            scores = outputs.logits.squeeze().cpu().tolist()
        
        # 为候选结果添加重排序分数
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = scores[i]
        
        # 按重排序分数排序
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # 限制结果数量
        if len(candidates) > config.RERANK_TOP_K:
            candidates = candidates[:config.RERANK_TOP_K]
        
        logger.debug(f"重排序完成，结果数: {len(candidates)}")
        return candidates
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        完整检索流程：混合检索 + 重排序
        
        Args:
            query: 查询文本
            
        Returns:
            最终检索结果
        """
        logger.info(f"执行检索，查询: {query}")
        
        # 1. 混合检索
        hybrid_results = self.hybrid_search(query)
        
        if not hybrid_results:
            logger.warning("混合检索未返回任何结果")
            return []
        
        # 2. 重排序
        final_results = self.rerank(query, hybrid_results)
        
        return final_results

# 创建高级检索器实例
advanced_retriever = AdvancedRetriever()
