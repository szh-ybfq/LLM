import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from config import config

def setup_logging():
    """配置系统日志，包括控制台和文件输出"""
    
    # 日志格式
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 根日志设置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除已存在的处理器
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    
    # 文件处理器 - 按大小轮转
    log_file = config.LOGS_DIR / "rag_system.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1024 * 1024 * 10,  # 10MB
        backupCount=10,  # 最多保留10个备份
        encoding="utf-8"
    )
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)
    
    # 第三方库日志级别调整，减少冗余输出
    for lib in ["urllib3", "huggingface_hub", "pymilvus"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("日志系统初始化完成")
    logger.info(f"日志文件路径: {log_file}")
    return logger

# 初始化日志
logger = setup_logging()
