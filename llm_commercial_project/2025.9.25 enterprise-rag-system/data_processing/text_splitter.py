import re
import nltk
from typing import List, Dict, Tuple
from nltk.tokenize import sent_tokenize
from config import config
from utils.logging import logger

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SmartTextSplitter:
    """智能文本分块器，基于语义和结构进行文本分块"""
    
    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        min_chunk_size: int = config.MIN_CHUNK_SIZE
    ):
        """
        初始化文本分块器
        
        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 块之间的重叠字符数
            min_chunk_size: 最小块大小，过滤过短的块
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        logger.info(f"文本分块器初始化完成，块大小: {chunk_size}, 重叠: {chunk_overlap}")
        
        # 段落分隔符模式
        self.paragraph_sep_pattern = re.compile(r'\n\s*\n', re.UNICODE)
        
        # 句子分割器
        self.sentence_tokenizer = sent_tokenize
    
    def split_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        将文本分割为多个块
        
        Args:
            text: 要分割的文本
            metadata: 文本的元数据，将被添加到每个块中
            
        Returns:
            块的列表，每个块包含内容和元数据
        """
        if not text or not text.strip():
            logger.warning("尝试分割空文本")
            return []
            
        metadata = metadata or {}
        
        # 首先按段落分割
        paragraphs = self._split_into_paragraphs(text)
        logger.debug(f"文本分割为 {len(paragraphs)} 个段落")
        
        # 处理段落，确保它们的大小合适
        chunks = self._process_paragraphs(paragraphs, metadata)
        
        # 过滤过短的块
        filtered_chunks = [chunk for chunk in chunks if len(chunk["content"]) >= self.min_chunk_size]
        logger.debug(f"文本分割完成，原始块数: {len(chunks)}, 过滤后块数: {len(filtered_chunks)}")
        
        return filtered_chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """将文本分割为段落"""
        paragraphs = self.paragraph_sep_pattern.split(text)
        # 清理段落（去除前后空白）
        paragraphs = [para.strip() for para in paragraphs if para.strip()]
        return paragraphs
    
    def _process_paragraphs(self, paragraphs: List[str], base_metadata: Dict) -> List[Dict]:
        """处理段落，将过长的段落分割为更小的块"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # 如果当前段落加上现有内容超过块大小
            if current_length + para_length > self.chunk_size and current_chunk:
                # 完成当前块
                chunk_content = ' '.join(current_chunk)
                chunks.append(self._create_chunk(chunk_content, base_metadata, len(chunks)))
                
                # 准备下一个块，保留重叠部分
                overlap_tokens = self._get_overlap_tokens(current_chunk)
                current_chunk = overlap_tokens + [para]
                current_length = len(' '.join(current_chunk))
            else:
                # 添加到当前块
                current_chunk.append(para)
                current_length += para_length
        
        # 添加最后一个块
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append(self._create_chunk(chunk_content, base_metadata, len(chunks)))
        
        return chunks
    
    def _get_overlap_tokens(self, current_chunk: List[str]) -> List[str]:
        """从当前块的末尾提取重叠部分"""
        chunk_text = ' '.join(current_chunk)
        sentences = self.sentence_tokenizer(chunk_text)
        
        # 计算需要保留的重叠句子
        overlap_text = ""
        overlap_length = 0
        overlap_sentences = []
        
        # 从后往前添加句子，直到达到重叠长度
        for sentence in reversed(sentences):
            sentence_length = len(sentence)
            if overlap_length + sentence_length <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += sentence_length
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk(self, content: str, base_metadata: Dict, chunk_index: int) -> Dict:
        """创建块字典，包含内容和元数据"""
        # 为块创建元数据（继承基础元数据并添加块特定信息）
        chunk_metadata = base_metadata.copy()
        chunk_metadata.update({
            "chunk_index": chunk_index,
            "chunk_length": len(content),
            "chunk_tokens": len(content.split())
        })
        
        return {
            "content": content,
            "metadata": chunk_metadata
        }

# 创建文本分块器实例
text_splitter = SmartTextSplitter()
