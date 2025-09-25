from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
from typing import List, Dict, Tuple, Optional
import json
from config import config
from utils.logging import logger

class MilvusClient:
    """Milvus向量数据库客户端，用于存储和检索向量嵌入"""
    
    def __init__(
        self,
        host: str = config.MILVUS_HOST,
        port: int = config.MILVUS_PORT,
        collection_name: str = config.MILVUS_COLLECTION
    ):
        """
        初始化Milvus客户端
        
        Args:
            host: Milvus服务器主机
            port: Milvus服务器端口
            collection_name: 集合名称
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = config.EMBEDDING_DIM
        
        # 连接到Milvus
        self._connect()
        
        # 确保集合存在
        self._ensure_collection()
        
        logger.info(f"Milvus客户端初始化完成，集合: {collection_name}")
    
    def _connect(self):
        """连接到Milvus服务器"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"成功连接到Milvus服务器: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接Milvus服务器失败: {str(e)}", exc_info=True)
            raise
    
    def _ensure_collection(self):
        """确保集合存在，如果不存在则创建"""
        if not utility.has_collection(self.collection_name):
            logger.info(f"集合 {self.collection_name} 不存在，创建新集合")
            
            # 定义字段
            fields = [
                # 主键，自动生成
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                # 向量字段
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                # 文本内容
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                # 元数据，存储为JSON字符串
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
            ]
            
            # 创建集合 schema
            schema = CollectionSchema(
                fields=fields,
                description="Enterprise RAG System collection"
            )
            
            # 创建集合
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            # 创建索引
            index_params = {
                "index_type": config.MILVUS_INDEX_TYPE,
                "metric_type": config.MILVUS_METRIC_TYPE,
                "params": {"M": 16, "efConstruction": 64}  # HNSW索引参数
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info(f"集合 {self.collection_name} 创建完成，并创建了索引")
        else:
            # 加载现有集合
            self.collection = Collection(self.collection_name)
            logger.info(f"加载现有集合: {self.collection_name}")
    
    def insert(self, data: List[Tuple[str, Dict, List[float]]]) -> List[int]:
        """
        插入数据到集合中
        
        Args:
            data: 数据列表，每个元素是 (content, metadata, embedding) 的元组
            
        Returns:
            插入的记录ID列表
        """
        if not data:
            logger.warning("尝试插入空数据")
            return []
        
        # 准备插入数据
        insert_data = []
        for content, metadata, embedding in data:
            # 确保内容和元数据不超过最大长度
            content_truncated = content[:65534] if len(content) > 65535 else content
            metadata_json = json.dumps(metadata)[:65534] if len(json.dumps(metadata)) > 65535 else json.dumps(metadata)
            
            insert_data.append({
                "content": content_truncated,
                "metadata": metadata_json,
                "embedding": embedding
            })
        
        # 执行插入
        try:
            result = self.collection.insert(insert_data)
            logger.info(f"成功插入 {len(data)} 条记录，IDs: {result.primary_keys}")
            
            # 刷新集合使数据可查
            self.collection.flush()
            
            return result.primary_keys
        except Exception as e:
            logger.error(f"插入数据失败: {str(e)}", exc_info=True)
            raise
    
    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filter_expr: Optional[str] = None,
        output_fields: List[str] = ["content", "metadata"]
    ) -> List[Dict]:
        """
        搜索相似向量
        
        Args:
            query_embedding: 查询向量
            limit: 返回结果数量
            filter_expr: 过滤表达式
            output_fields: 需要返回的字段
            
        Returns:
            搜索结果列表，包含内容、元数据和相似度分数
        """
        if not query_embedding:
            raise ValueError("查询向量不能为空")
        
        # 加载集合（如果尚未加载）
        self.collection.load()
        
        # 准备搜索参数
        search_params = {
            "metric_type": config.MILVUS_METRIC_TYPE,
            "params": {"ef": 64}  # 搜索时的参数
        }
        
        # 执行搜索
        try:
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=filter_expr,
                output_fields=output_fields
            )
            
            # 处理搜索结果
            processed_results = []
            for hit in results[0]:  # 因为我们只搜索了一个向量
                # 解析元数据
                metadata = json.loads(hit.entity.get("metadata")) if hit.entity.get("metadata") else {}
                
                processed_results.append({
                    "id": hit.id,
                    "content": hit.entity.get("content"),
                    "metadata": metadata,
                    "score": hit.score
                })
            
            logger.debug(f"搜索完成，返回 {len(processed_results)} 条结果")
            return processed_results
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}", exc_info=True)
            raise
    
    def count(self) -> int:
        """返回集合中的记录数量"""
        return self.collection.num_entities
    
    def delete(self, expr: str) -> None:
        """
        根据表达式删除记录
        
        Args:
            expr: 删除条件表达式
        """
        try:
            result = self.collection.delete(expr)
            logger.info(f"删除记录数量: {result.delete_count}")
            self.collection.flush()
        except Exception as e:
            logger.error(f"删除记录失败: {str(e)}", exc_info=True)
            raise
    
    def drop_collection(self) -> None:
        """删除整个集合（谨慎使用）"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.warning(f"集合 {self.collection_name} 已删除")
        else:
            logger.warning(f"集合 {self.collection_name} 不存在，无需删除")

# 创建Milvus客户端实例
milvus_client = MilvusClient()
