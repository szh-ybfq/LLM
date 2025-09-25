import argparse
import time
from config import config
from utils.logging import logger
from rag_pipeline import rag_pipeline

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="企业级RAG系统")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 摄入文档命令
    ingest_parser = subparsers.add_parser("ingest", help="摄入文档")
    ingest_parser.add_argument(
        "path", 
        help="文档路径或目录路径"
    )
    
    # 查询命令
    query_parser = subparsers.add_parser("query", help="查询")
    query_parser.add_argument(
        "text", 
        help="查询文本"
    )
    
    # 统计命令
    subparsers.add_parser("stats", help="显示系统统计信息")
    
    # 清除数据命令
    clear_parser = subparsers.add_parser("clear", help="清除所有数据")
    clear_parser.add_argument(
        "--force", 
        action="store_true", 
        help="强制清除，无需确认"
    )
    
    # 启动API服务命令
    api_parser = subparsers.add_parser("api", help="启动API服务")
    api_parser.add_argument(
        "--host", 
        default=config.API_HOST, 
        help=f"API主机地址，默认: {config.API_HOST}"
    )
    api_parser.add_argument(
        "--port", 
        type=int, 
        default=config.API_PORT, 
        help=f"API端口，默认: {config.API_PORT}"
    )
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        # 摄入文档
        start_time = time.time()
        try:
            if os.path.isfile(args.path):
                # 摄入单个文档
                total_chunks, inserted_chunks = rag_pipeline.ingest_document(args.path)
                logger.info(
                    f"文档摄入完成，总块数: {total_chunks}, 插入块数: {inserted_chunks}, "
                    f"耗时: {time.time() - start_time:.2f}秒"
                )
            elif os.path.isdir(args.path):
                # 摄入目录
                stats = rag_pipeline.ingest_directory(args.path)
                logger.info(
                    f"目录摄入完成，总文档数: {stats['total_documents']}, "
                    f"成功文档数: {stats['successful_documents']}, "
                    f"总块数: {stats['total_chunks']}, "
                    f"插入块数: {stats['inserted_chunks']}, "
                    f"耗时: {time.time() - start_time:.2f}秒"
                )
            else:
                logger.error(f"路径不存在或不是文件/目录: {args.path}")
        except Exception as e:
            logger.error(f"摄入失败: {str(e)}", exc_info=True)
    
    elif args.command == "query":
        # 执行查询
        start_time = time.time()
        try:
            result = rag_pipeline.query(args.text)
            logger.info(
                f"查询完成，生成token数: {result['generated_tokens']}, "
                f"耗时: {time.time() - start_time:.2f}秒"
            )
            print("\n查询:", result["query"])
            print("\n回答:", result["answer"])
            print("\n使用的上下文:")
            for i, context in enumerate(result["contexts"], 1):
                source = context["metadata"].get("file_name", "未知来源")
                print(f"\n上下文 {i} (来源: {source}):")
                print(context["content"][:200] + ("..." if len(context["content"]) > 200 else ""))
        except Exception as e:
            logger.error(f"查询失败: {str(e)}", exc_info=True)
    
    elif args.command == "stats":
        # 显示统计信息
        stats = rag_pipeline.get_stats()
        print("系统统计信息:")
        print(f"文档数量: {stats['document_count']}")
        print(f"块数量: {stats['chunk_count']}")
        print(f"嵌入模型: {stats['embedding_model']}")
        print(f"LLM模型: {stats['llm_model']}")
        print(f"向量数据库: {stats['vector_db']['name']} ({stats['vector_db']['host']}:{stats['vector_db']['port']})")
        print(f"设备: {stats['device']}")
    
    elif args.command == "clear":
        # 清除数据
        if args.force or input("确定要清除所有数据吗? (y/N) ").lower() == "y":
            try:
                rag_pipeline.clear_data()
                print("所有数据已清除")
            except Exception as e:
                logger.error(f"清除数据失败: {str(e)}", exc_info=True)
        else:
            print("取消清除操作")
    
    elif args.command == "api":
        # 启动API服务
        from api.main import app
        import uvicorn
        logger.info(f"启动API服务，地址: http://{args.host}:{args.port}")
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            workers=config.API_WORKERS,
            reload=config.DEBUG
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    import os
    main()
