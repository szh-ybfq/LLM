from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import shutil
from config import config
from utils.logging import logger
from rag_pipeline import rag_pipeline

# 初始化FastAPI应用
app = FastAPI(
    title=config.PROJECT_NAME,
    version=config.VERSION,
    description="企业级RAG系统API服务"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型
class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str

class QueryResponse(BaseModel):
    """查询响应模型"""
    query: str
    answer: str
    contexts: List[Dict]
    generated_tokens: int
    retrieved_contexts_count: int

class IngestResponse(BaseModel):
    """文档摄入响应模型"""
    total_documents: int
    successful_documents: int
    total_chunks: int
    inserted_chunks: int
    failed_documents: List[Dict]

class StatsResponse(BaseModel):
    """系统统计响应模型"""
    document_count: int
    chunk_count: int
    embedding_model: str
    llm_model: str
    vector_db: Dict[str, Any]
    device: str

# API路由
@app.get("/", tags=["根路径"])
async def root():
    """根路径，返回系统信息"""
    return {
        "name": config.PROJECT_NAME,
        "version": config.VERSION,
        "status": "running"
    }

@app.post("/query", tags=["查询"], response_model=QueryResponse)
async def query(request: QueryRequest):
    """处理用户查询，返回生成的回答"""
    try:
        logger.info(f"收到查询: {request.query}")
        result = rag_pipeline.query(request.query)
        return result
    except Exception as e:
        logger.error(f"查询处理错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询处理错误: {str(e)}")

@app.post("/ingest/document", tags=["文档摄入"])
async def ingest_document(file: UploadFile = File(...)):
    """摄入单个文档"""
    try:
        # 创建临时目录
        temp_dir = config.DATA_DIR / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存上传的文件
        file_path = temp_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 摄入文档
        total_chunks, inserted_chunks = rag_pipeline.ingest_document(str(file_path))
        
        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return {
            "file_name": file.filename,
            "total_chunks": total_chunks,
            "inserted_chunks": inserted_chunks,
            "status": "success" if inserted_chunks > 0 else "failed"
        }
    except Exception as e:
        logger.error(f"文档摄入错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档摄入错误: {str(e)}")

@app.post("/ingest/directory", tags=["文档摄入"], response_model=IngestResponse)
async def ingest_directory(directory_path: str = Form(...)):
    """摄入目录中的所有文档"""
    try:
        if not os.path.isdir(directory_path):
            raise HTTPException(status_code=400, detail=f"目录不存在: {directory_path}")
        
        logger.info(f"开始摄入目录: {directory_path}")
        stats = rag_pipeline.ingest_directory(directory_path)
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"目录摄入错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"目录摄入错误: {str(e)}")

@app.get("/stats", tags=["系统信息"], response_model=StatsResponse)
async def get_stats():
    """获取系统统计信息"""
    try:
        stats = rag_pipeline.get_stats()
        return stats
    except Exception as e:
        logger.error(f"获取统计信息错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取统计信息错误: {str(e)}")

@app.delete("/data", tags=["数据管理"])
async def clear_data(confirm: bool = False):
    """清除所有数据（谨慎使用）"""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="请设置confirm=true以确认清除所有数据"
        )
    
    try:
        rag_pipeline.clear_data()
        return {"status": "success", "message": "所有数据已清除"}
    except Exception as e:
        logger.error(f"清除数据错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"清除数据错误: {str(e)}")

# 启动服务
if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动API服务，地址: http://{config.API_HOST}:{config.API_PORT}")
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=config.DEBUG
    )
