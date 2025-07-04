"""
Multi-View Recall System
========================
实现多视角召回，包括：
1. Tag → Tag 精确匹配（倒排索引）
2. Tag → Title/Intro 语义检索（向量检索）
3. 补充规则召回（同IP、热门故事）
"""

import re
import json
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logger.warning("sentence-transformers not installed")

try:
    from whoosh.index import create_in
    from whoosh.fields import *
    from whoosh.qparser import QueryParser
    from whoosh.analysis import StandardAnalyzer
except ImportError:
    create_in = None
    logger.warning("whoosh not installed, falling back to simple tag matching")


class MultiViewRecall:
    """多视角召回系统"""
    
    def __init__(self, contents_df: pd.DataFrame, interactions_df: Optional[pd.DataFrame] = None):
        self.contents_df = contents_df
        self.interactions_df = interactions_df
        
        # 构建内容标签索引
        self._build_content_tags()
        
        # 构建倒排索引
        self._build_inverted_index()
        
        # 构建向量索引
        self._build_vector_index()
        
        # 构建热门内容索引
        self._build_popularity_index()
        
        self.popular_weight = 0.25
        
        logger.info(f"MultiViewRecall initialized - {len(contents_df)} contents, {len(self.content_tags)} tagged")
    
    def _build_content_tags(self):
        """为每条内容提取标签"""
        self.content_tags = {}
        self.tag_to_contents = defaultdict(set)
        
        for _, row in self.contents_df.iterrows():
            content_id = int(row['content_id'])
            title = str(row.get('title', '')).lower()
            intro = str(row.get('intro', '')).lower()
            
            # 提取标签（从标题和简介中）
            tags = self._extract_tags_from_text(title + ' ' + intro)
            self.content_tags[content_id] = tags
            
            # 构建倒排索引
            for tag in tags:
                self.tag_to_contents[tag].add(content_id)
        
        logger.info(f"Extracted {len(self.tag_to_contents)} unique tags")
    
    def _extract_tags_from_text(self, text: str) -> Set[str]:
        """从文本中提取标签"""
        # 预定义的标签列表
        predefined_tags = {
            # 热门IP
            'naruto', 'one piece', 'my hero academia', 'demon slayer', 'jujutsu kaisen',
            'attack on titan', 'sword art online', 'genshin impact', 'pokemon',
            'dragon ball', 'bleach', 'fairy tail', 'high school dxd',
            
            # 类型标签
            'romance', 'action', 'comedy', 'drama', 'fantasy', 'slice of life',
            'adventure', 'mystery', 'horror', 'sci-fi', 'isekai', 'harem',
            'reverse harem', 'yandere', 'tsundere', 'kuudere', 'yuri', 'yaoi',
            
            # 主题标签
            'school', 'supernatural', 'magic', 'superpower', 'vampire', 'werewolf',
            'mafia', 'office', 'family', 'childhood friends', 'enemies to lovers',
            'love triangle', 'forbidden love', 'obsession', 'protective',
            
            # 角色类型
            'original character', 'self-insert', 'villain', 'anti-hero',
            'mentor', 'student', 'teacher', 'boss', 'employee',
            
            # 新增标签
            'obsession', 'protective', 'childhood friends', 'office romance', 'fake relationship', 'slow burn',
        }
        
        # 从文本中匹配标签
        found_tags = set()
        text_lower = text.lower()
        
        for tag in predefined_tags:
            if tag in text_lower:
                found_tags.add(tag)
        
        # 提取一些特殊模式
        patterns = [
            r'(\w+)\s+fan',  # xxx fan
            r'(\w+)\s+romance',  # xxx romance
            r'(\w+)\s+au',  # xxx au
            r'(\w+)\s+crossover',  # xxx crossover
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            found_tags.update(matches)
        
        return found_tags
    
    def _build_inverted_index(self):
        """构建倒排索引（如果whoosh可用）"""
        if create_in is None:
            logger.info("Using simple tag matching (whoosh not available)")
            return
        
        try:
            # 创建whoosh索引
            schema = Schema(
                content_id=ID(stored=True),
                title=TEXT(stored=True),
                intro=TEXT(stored=True),
                tags=TEXT(stored=True)
            )
            
            import tempfile
            import os
            temp_dir = tempfile.mkdtemp()
            self.whoosh_index = create_in(temp_dir, schema)
            self.writer = self.whoosh_index.writer()
            
            # 添加文档
            for content_id, tags in self.content_tags.items():
                row = self.contents_df[self.contents_df['content_id'] == content_id].iloc[0]
                self.writer.add_document(
                    content_id=str(content_id),
                    title=str(row.get('title', '')),
                    intro=str(row.get('intro', '')),
                    tags=' '.join(tags)
                )
            
            self.writer.commit()
            logger.info("Whoosh inverted index built successfully")
            
        except Exception as e:
            logger.warning(f"Failed to build whoosh index: {e}")
            self.whoosh_index = None
    
    def _build_vector_index(self):
        """构建向量索引"""
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not available, skipping vector index")
            self.vector_model = None
            return
        
        try:
            self.vector_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 准备文本
            texts = []
            content_ids = []
            for _, row in self.contents_df.iterrows():
                content_id = int(row['content_id'])
                title = str(row.get('title', ''))
                intro = str(row.get('intro', ''))
                text = f"{title} {intro}".strip()
                if text:
                    texts.append(text)
                    content_ids.append(content_id)
            
            if texts:
                self.content_embeddings = self.vector_model.encode(texts, show_progress_bar=False)
                self.embedding_content_ids = content_ids
                logger.info(f"Vector index built - {len(texts)} embeddings")
            else:
                logger.warning("No valid texts for vector indexing")
                
        except Exception as e:
            logger.warning(f"Failed to build vector index: {e}")
            self.vector_model = None
    
    def _build_popularity_index(self):
        """构建热门内容索引"""
        self.popular_contents = []
        
        if self.interactions_df is not None and not self.interactions_df.empty:
            popular = self.interactions_df.sort_values('interaction_count', ascending=False)
            self.popular_contents = popular["content_id"].astype(int).tolist()[:100]
            logger.info(f"Popularity index built - {len(self.popular_contents)} popular contents")
        else:
            # 如果没有交互数据，随机选择一些作为"热门"
            self.popular_contents = self.contents_df['content_id'].sample(min(50, len(self.contents_df))).tolist()
            logger.info(f"Popularity index built (random) - {len(self.popular_contents)} contents")
    
    def recall(self, user_tags: List[str], top_k: int = 120) -> List[Tuple[int, float, str]]:
        """
        多视角召回
        
        Args:
            user_tags: 用户标签
            top_k: 召回数量
            
        Returns:
            List of (content_id, score, method)
        """
        results = []
        seen_content_ids = set()
        
        # 1. Tag → Tag 精确匹配
        tag_matches = self._tag_precise_match(user_tags)
        for content_id, score in tag_matches:
            if content_id not in seen_content_ids:
                results.append((content_id, score, 'tag_match'))
                seen_content_ids.add(content_id)
        
        # 2. Tag → Title/Intro 语义检索
        semantic_matches = self._semantic_search(user_tags, top_k=300)
        for content_id, score in semantic_matches:
            if content_id not in seen_content_ids:
                results.append((content_id, score, 'semantic'))
                seen_content_ids.add(content_id)
        
        # 3. 补充规则召回
        rule_matches = self._rule_based_recall(user_tags, top_k=100)
        for content_id, score in rule_matches:
            if content_id not in seen_content_ids:
                results.append((content_id, score, 'rule'))
                seen_content_ids.add(content_id)
        
        # 4. 热门内容缺位补齐
        top_k_after_tag_semantic = len([r for r in results if r[2] in ['tag_match', 'semantic']])
        if len(results) < top_k:
            popular_matches = self._popularity_recall(top_k=top_k - len(results))
            for content_id, score in popular_matches:
                if content_id not in seen_content_ids:
                    results.append((content_id, score, 'popular'))
                    seen_content_ids.add(content_id)
        
        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Multi-view recall: {len(results)} candidates "
                   f"(tag: {len([r for r in results if r[2]=='tag_match'])}, "
                   f"semantic: {len([r for r in results if r[2]=='semantic'])}, "
                   f"rule: {len([r for r in results if r[2]=='rule'])}, "
                   f"popular: {len([r for r in results if r[2]=='popular'])})")
        
        print("---SNAPSHOT------------------------------------------------")
        print("#candidates  :", len(results))
        print("example IDs :", [cid for cid,_,_ in results[:15]])
        return results[:top_k]
    
    def _tag_precise_match(self, user_tags: List[str]) -> List[Tuple[int, float]]:
        """Tag → Tag 精确匹配"""
        matches = []
        user_tag_set = set(user_tags)
        
        for content_id, content_tags in self.content_tags.items():
            overlap = len(user_tag_set & content_tags)
            if overlap > 0:
                # 分数 = 重叠标签数 / 用户标签数
                score = overlap / len(user_tag_set)
                matches.append((content_id, score))
        
        # 按分数排序
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:200]  # 限制数量
    
    def _semantic_search(self, user_tags: List[str], top_k: int = 300) -> List[Tuple[int, float]]:
        """Tag → Title/Intro 语义检索"""
        if self.vector_model is None or self.content_embeddings is None:
            return []
        
        try:
            # 将用户标签转换为查询文本
            query_text = ' '.join(user_tags)
            query_embedding = self.vector_model.encode([query_text])[0]
            
            # 计算相似度
            similarities = np.dot(self.content_embeddings, query_embedding) / (
                np.linalg.norm(self.content_embeddings, axis=1) * 
                np.linalg.norm(query_embedding) + 1e-8
            )
            
            # 获取top_k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            matches = []
            
            for idx in top_indices:
                content_id = self.embedding_content_ids[idx]
                score = float(similarities[idx])
                matches.append((content_id, score))
            
            return matches
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []
    
    def _rule_based_recall(self, user_tags: List[str], top_k: int = 100) -> List[Tuple[int, float]]:
        """补充规则召回"""
        matches = []
        
        # 同IP召回
        for content_id, content_tags in self.content_tags.items():
            score = 0.0
            
            # 检查是否有相同IP
            for user_tag in user_tags:
                if user_tag in content_tags:
                    # 找到同IP的其他内容
                    for other_id, other_tags in self.content_tags.items():
                        if other_id != content_id and user_tag in other_tags:
                            # 计算相似度
                            similarity = len(content_tags & other_tags) / max(len(content_tags | other_tags), 1)
                            score += similarity * 0.3  # 权重
            
            if score > 0:
                matches.append((content_id, score))
        
        # 按分数排序
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]
    
    def _popularity_recall(self, top_k: int = 50) -> List[Tuple[int, float]]:
        """热门内容召回"""
        results = []
        for i, cid in enumerate(self.popular_contents[:top_k]):
            # 原分数: 1.0 - (i / len(self.popular_contents)) * 0.5
            score = (1.0 - (i / len(self.popular_contents))) * self.popular_weight
            results.append((cid, score, 'popular'))
        return results
    
    def get_content_tags(self, content_id: int) -> Set[str]:
        """获取内容的标签"""
        return self.content_tags.get(content_id, set())
    
    def get_tag_statistics(self) -> Dict[str, int]:
        """获取标签统计"""
        tag_counts = Counter()
        for tags in self.content_tags.values():
            tag_counts.update(tags)
        return dict(tag_counts.most_common(50))
