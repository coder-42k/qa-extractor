"""Semantic Q&A Detection Module - Hybrid Rule-based and Embedding-based Detection"""

import re
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SemanticQADetector:
    """
    混合式QA检测器：结合规则匹配和语义相似度
    
    核心创新：
    1. 高速规则预筛选 - 保持高效率的基线性能
    2. 语义相似度验证 - 提升泛化能力和召回率
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 similarity_threshold: float = 0.65,
                 potential_question_max_length: int = 50):
        """
        初始化语义QA检测器
        
        Args:
            model_name: 句子嵌入模型名称
            similarity_threshold: 语义相似度阈值
            potential_question_max_length: 潜在问题的最大长度
        """
        self.logger = logger
        self.similarity_threshold = similarity_threshold
        self.potential_question_max_length = potential_question_max_length
        
        # 初始化句子嵌入模型
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
        # 规则模式定义（复用现有模式）
        self.direct_question_patterns = [
            r"网友[：:]", r"问[：:]", r"问题[：:]", r"提问[：:]",
            r"主持人[：:]", r"观众[：:]", r"Q[：:]", r"记者[：:]"
        ]
        
        self.answer_patterns = [
            r"段永平[：:]", r"段[：:]", r"大道[：:]", r"答[：:]", r"A[：:]"
        ]
        
        # 疑问语气模式
        self.question_indicators = [
            r"[？?]$", r"吗[？?。]?$", r"呢[？?。]?$", r"么[？?。]?$",
            r"怎么", r"什么", r"为什么", r"如何", r"是否", r"能否",
            r"有没有", r"会不会", r"可以吗", r"对吗", r"是吗"
        ]
    
    def detect_qa_pairs(self, paragraphs: List[str]) -> List[Tuple[int, int]]:
        """
        检测文本中的QA对边界
        
        Args:
            paragraphs: 段落列表
            
        Returns:
            QA对的起止索引列表 [(start_idx, end_idx), ...]
        """
        qa_pairs = []
        
        # 第一层：规则预筛选
        rule_based_pairs = self._rule_based_detection(paragraphs)
        qa_pairs.extend(rule_based_pairs)
        
        # 第二层：语义相似度验证（如果嵌入模型可用）
        if self.embedding_model:
            semantic_pairs = self._semantic_detection(paragraphs, rule_based_pairs)
            qa_pairs.extend(semantic_pairs)
        else:
            self.logger.warning("Embedding model not available, using rule-based detection only")
        
        # 合并重叠的QA对
        qa_pairs = self._merge_overlapping_pairs(qa_pairs)
        
        return qa_pairs
    
    def _rule_based_detection(self, paragraphs: List[str]) -> List[Tuple[int, int]]:
        """基于规则的QA检测（高速预筛选）"""
        qa_pairs = []
        i = 0
        
        while i < len(paragraphs):
            # 检查是否是问题开始
            if self._is_question_by_rule(paragraphs[i]):
                start_idx = i
                # 查找对应的答案结束位置
                end_idx = self._find_answer_end(paragraphs, i + 1)
                if end_idx > start_idx:
                    qa_pairs.append((start_idx, end_idx))
                    i = end_idx + 1
                else:
                    i += 1
            else:
                i += 1
        
        return qa_pairs
    
    def _semantic_detection(self, paragraphs: List[str], 
                          existing_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """基于语义相似度的QA检测（提升泛化能力）"""
        semantic_pairs = []
        
        # 获取已被规则识别的索引
        covered_indices = set()
        for start, end in existing_pairs:
            covered_indices.update(range(start, end + 1))
        
        # 查找潜在问题
        potential_questions = []
        for i, para in enumerate(paragraphs):
            if i not in covered_indices and self._is_potential_question(para):
                potential_questions.append(i)
        
        # 对每个潜在问题，计算与后续段落的语义相似度
        for q_idx in potential_questions:
            if q_idx + 1 >= len(paragraphs):
                continue
                
            # 获取问题和候选答案的嵌入
            question_embedding = self.embedding_model.encode(paragraphs[q_idx])
            
            # 计算与后续段落的相似度
            max_similarity = 0
            best_end_idx = q_idx
            
            for j in range(q_idx + 1, min(q_idx + 5, len(paragraphs))):  # 最多检查后续4个段落
                if j in covered_indices:
                    break
                    
                answer_embedding = self.embedding_model.encode(paragraphs[j])
                similarity = self._cosine_similarity(question_embedding, answer_embedding)
                
                if similarity > self.similarity_threshold and similarity > max_similarity:
                    max_similarity = similarity
                    best_end_idx = j
            
            # 如果找到语义相关的答案，添加为QA对
            if best_end_idx > q_idx:
                semantic_pairs.append((q_idx, best_end_idx))
                self.logger.debug(f"Found semantic QA pair: Q[{q_idx}] -> A[{q_idx+1}:{best_end_idx}], similarity: {max_similarity:.3f}")
        
        return semantic_pairs
    
    def _is_question_by_rule(self, text: str) -> bool:
        """基于规则判断是否为问题"""
        return any(re.search(pattern, text) for pattern in self.direct_question_patterns)
    
    def _is_potential_question(self, text: str) -> bool:
        """判断是否为潜在问题（无明显前缀但有疑问语气）"""
        # 长度检查
        if len(text.strip()) > self.potential_question_max_length:
            return False
        
        # 疑问语气检查
        return any(re.search(pattern, text) for pattern in self.question_indicators)
    
    def _find_answer_end(self, paragraphs: List[str], start_idx: int) -> int:
        """查找答案的结束位置"""
        if start_idx >= len(paragraphs):
            return start_idx - 1
        
        # 检查是否有答案标记
        for i in range(start_idx, len(paragraphs)):
            # 如果遇到新的问题，则前一段是答案结束
            if self._is_question_by_rule(paragraphs[i]):
                return i - 1
            
            # 如果有明确的答案标记
            if any(re.search(pattern, paragraphs[i]) for pattern in self.answer_patterns):
                # 继续查找直到下一个问题或段落结束
                for j in range(i + 1, len(paragraphs)):
                    if self._is_question_by_rule(paragraphs[j]):
                        return j - 1
                return len(paragraphs) - 1
        
        # 默认下一段为答案
        return min(start_idx, len(paragraphs) - 1)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _merge_overlapping_pairs(self, pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """合并重叠的QA对"""
        if not pairs:
            return []
        
        # 按起始位置排序
        sorted_pairs = sorted(pairs)
        merged = [sorted_pairs[0]]
        
        for current in sorted_pairs[1:]:
            last = merged[-1]
            # 如果有重叠，合并
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    def showcase_demo(self, text: str) -> Dict[str, any]:
        """
        展示demo：识别无明显前缀的问答对
        
        Returns:
            包含检测结果和可视化信息的字典
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # 分别进行规则检测和语义检测
        rule_pairs = self._rule_based_detection(paragraphs)
        semantic_pairs = [] if not self.embedding_model else self._semantic_detection(paragraphs, rule_pairs)
        
        result = {
            'paragraphs': paragraphs,
            'rule_based_pairs': rule_pairs,
            'semantic_pairs': semantic_pairs,
            'all_pairs': self._merge_overlapping_pairs(rule_pairs + semantic_pairs),
            'showcase_message': self._generate_showcase_message(rule_pairs, semantic_pairs)
        }
        
        return result
    
    def _generate_showcase_message(self, rule_pairs: List, semantic_pairs: List) -> str:
        """生成展示信息"""
        if not semantic_pairs:
            return "仅使用规则检测"
        
        return f"""
        🎯 混合式检测结果：
        - 规则检测：发现 {len(rule_pairs)} 个QA对
        - 语义检测：额外发现 {len(semantic_pairs)} 个隐含QA对
        - 技术亮点：成功识别出无明显前缀的问答对，展示了语义理解能力
        """