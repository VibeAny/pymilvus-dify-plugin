# Milvus RAG 策略指南

## 概述

本文档记录了在使用Milvus向量数据库构建RAG（检索增强生成）系统时的最佳实践和策略建议。

## 1. 文档切分(Chunking)策略

### 1.1 切分方法对比

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **固定长度切分** | 简单、一致 | 可能切断语义 | 结构化文档 |
| **段落边界切分** | 保持语义完整 | 长度不均匀 | 文章、博客 |
| **句子边界切分** | 语义清晰 | 可能太短 | 问答文档 |
| **混合策略** | 兼顾完整性和一致性 | 复杂度高 | 通用场景 |

### 1.2 推荐切分配置

#### 宠物护理文档（当前场景）
```python
配置参数：
- 主要边界：段落分隔符 (\n\n)
- 最大长度：500-600字符
- 重叠长度：50-100字符
- 最小长度：100字符（过滤过短chunk）
- 保持：完整句子不被切断
```

#### 技术文档
```python
配置参数：
- 主要边界：章节标题 (#, ##, ###)
- 最大长度：800-1000字符
- 重叠长度：100-150字符
- 特殊处理：代码块保持完整
```

#### FAQ文档
```python
配置参数：
- 边界：问答对 (Q&A pairs)
- 策略：每个问答对一个chunk
- 格式：问题+答案保持在一起
```

### 1.3 切分实现示例

```python
def smart_chunking(text, max_chunk_size=600, overlap=100, min_chunk_size=100):
    """
    智能文档切分算法
    
    Args:
        text: 输入文本
        max_chunk_size: 最大chunk长度
        overlap: 重叠字符数
        min_chunk_size: 最小chunk长度
    
    Returns:
        List[str]: 切分后的chunks
    """
    # 1. 按段落初步切分
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # 2. 处理超长段落
        if len(paragraph) > max_chunk_size:
            # 按句子再次切分
            sentences = re.split(r'[.!?。！？]+', paragraph)
            for sentence in sentences:
                if len(current_chunk + sentence) > max_chunk_size:
                    if len(current_chunk) >= min_chunk_size:
                        chunks.append(current_chunk.strip())
                        # 保持重叠
                        current_chunk = current_chunk[-overlap:] + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk += sentence
        else:
            # 3. 正常段落处理
            if len(current_chunk + paragraph) > max_chunk_size:
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = current_chunk[-overlap:] + paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    # 添加最后一个chunk
    if len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk.strip())
    
    return chunks
```

## 2. 向量检索策略

### 2.1 召回(Recall)基础概念

召回是信息检索系统的核心指标，决定了系统能找到多少相关信息。

#### 召回率 vs 精确率
```
召回率(Recall) = 检索到的相关文档数 / 数据库中所有相关文档数
精确率(Precision) = 检索到的相关文档数 / 检索到的总文档数

示例：
- 数据库中有100个关于"狗狗护理"的文档
- 用户查询"如何给狗狗洗澡"
- 系统返回了10个结果，其中8个是相关的
- 但数据库中实际有15个相关文档

召回率 = 8/15 = 53.3%（错过了7个相关文档）
精确率 = 8/10 = 80%（返回的结果中80%是相关的）
```

#### 近似 vs 精确召回
```python
# HNSW索引参数对召回率的影响
{
    "index_type": "HNSW",
    "params": {
        "M": 16,
        "efConstruction": 200,
        "ef": 64  # 关键参数，直接影响召回率
    }
}

# ef参数与性能的权衡：
- ef=64:  快速检索，召回率~85%，适合实时查询
- ef=128: 中等速度，召回率~92%，推荐的平衡点
- ef=256: 较慢检索，召回率~97%，适合离线分析
```

### 2.2 检索策略对比

| 策略 | 描述 | 优点 | 缺点 | 推荐场景 |
|------|------|------|------|----------|
| **单一最佳** | 只返回相似度最高的1个chunk | 精确、快速 | 上下文不足，召回率低 | 简单问答 |
| **Top-K固定** | 返回固定数量的top结果 | 稳定、可预期 | 可能包含噪音 | 一般查询 |
| **阈值过滤** | 返回相似度>阈值的所有结果 | 质量保证 | 数量不稳定 | 质量优先 |
| **多路召回** | 向量+关键词+语义多路融合 | 高召回率，覆盖全面 | 复杂度高，需要去重 | 生产环境 |
| **混合策略** | Top-K + 阈值 + 智能筛选 | 平衡质量和数量 | 逻辑复杂 | 推荐方案 |

### 2.2 推荐检索配置

#### 基础配置
```yaml
搜索参数：
- 向量字段：embedding
- 相似度计算：COSINE（推荐用于文本）
- 搜索范围：limit=10（初始检索）
- 最小相似度：0.3（过滤低质量结果）
```

#### 返回字段配置
```yaml
output_fields：
- uuid: 文档唯一标识
- title: 文档标题
- path: 文件路径
- url: 文档链接
注意：不返回embedding字段（1536维，太大）
```

### 2.3 智能检索策略

```python
def intelligent_retrieval(search_results, query_type="general"):
    """
    智能检索结果筛选
    
    Args:
        search_results: Milvus检索原始结果
        query_type: 查询类型 (simple/general/complex/comprehensive)
    
    Returns:
        List: 筛选后的最优结果
    """
    
    if not search_results:
        return []
    
    best_similarity = search_results[0]['similarity']
    
    if query_type == "simple":
        # 简单事实查询：只要最佳的1-2个chunk
        threshold = 0.6
        return [r for r in search_results[:2] if r['similarity'] >= threshold]
    
    elif query_type == "general":
        # 一般查询：动态调整数量
        if best_similarity > 0.8:
            return search_results[:2]
        elif best_similarity > 0.6:
            return [r for r in search_results[:3] if r['similarity'] >= 0.5]
        else:
            return [r for r in search_results[:5] if r['similarity'] >= 0.4]
    
    elif query_type == "complex":
        # 复杂分析：需要更多上下文
        threshold = 0.4
        return [r for r in search_results[:7] if r['similarity'] >= threshold]
    
    elif query_type == "comprehensive":
        # 全面信息：按文档聚合
        doc_groups = {}
        for result in search_results:
            doc_path = result['entity']['path']
            if (doc_path not in doc_groups or 
                result['similarity'] > doc_groups[doc_path]['similarity']):
                doc_groups[doc_path] = result
        
        # 返回每个相关文档的最佳chunk
        return sorted(doc_groups.values(), 
                     key=lambda x: x['similarity'], reverse=True)[:4]
    
    return search_results[:3]  # 默认返回top 3
```

### 2.4 召回率优化策略

#### 多路召回实现
```python
def multi_path_recall(query_text, query_vector, collection_name):
    """
    多路召回策略，提高覆盖率
    
    Returns:
        List: 融合后的检索结果
    """
    all_results = []
    
    # 路径1：向量相似度召回（主要路径）
    vector_results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        anns_field="embedding",
        limit=50,  # 初始召回数量大一些
        search_params={"ef": 128},  # 提高召回率
        output_fields=["uuid", "title", "content", "path", "url"]
    )
    all_results.extend([(r, "vector", r['score']) for r in vector_results])
    
    # 路径2：关键词匹配召回（补充路径）
    keywords = extract_keywords(query_text)
    if keywords:
        keyword_filter = f"title like '%{keywords[0]}%'"
        keyword_results = client.query(
            collection_name=collection_name,
            filter=keyword_filter,
            output_fields=["uuid", "title", "content", "path", "url"],
            limit=20
        )
        # 为关键词结果设置基础分数
        all_results.extend([(r, "keyword", 0.5) for r in keyword_results])
    
    # 路径3：标题完全匹配召回（高精度路径）
    title_filter = f"title == '{query_text}'"
    try:
        title_results = client.query(
            collection_name=collection_name,
            filter=title_filter,
            output_fields=["uuid", "title", "content", "path", "url"],
            limit=5
        )
        # 标题匹配给予高分
        all_results.extend([(r, "title", 0.9) for r in title_results])
    except:
        pass  # 如果精确匹配失败，忽略此路径
    
    # 去重和融合
    return merge_and_dedupe_results(all_results)

def merge_and_dedupe_results(all_results, max_results=10):
    """融合多路召回结果并去重"""
    
    # 按uuid去重，保留最高分数
    uuid_to_result = {}
    for result, source, score in all_results:
        uuid = result.get('uuid')
        if uuid:
            if uuid not in uuid_to_result or score > uuid_to_result[uuid][2]:
                uuid_to_result[uuid] = (result, source, score)
    
    # 按分数排序
    sorted_results = sorted(uuid_to_result.values(), 
                          key=lambda x: x[2], reverse=True)
    
    # 返回最终结果
    return [{'entity': r[0], 'source': r[1], 'similarity': r[2]} 
            for r in sorted_results[:max_results]]

def extract_keywords(text, max_keywords=3):
    """提取查询文本的关键词"""
    import re
    
    # 简单的关键词提取（可以用更复杂的NLP方法）
    words = re.findall(r'\b\w+\b', text.lower())
    
    # 过滤停用词
    stop_words = {'的', '了', '在', '是', '有', '和', '就', '不', '人', '都', 
                  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    keywords = [w for w in words if len(w) > 1 and w not in stop_words]
    
    return keywords[:max_keywords]
```

#### 查询扩展策略
```python
def query_expansion_recall(original_query, collection_name):
    """通过查询扩展提高召回率"""
    
    # 生成原始查询的embedding
    original_vector = get_embedding(original_query)
    all_results = []
    
    # 1. 原始查询
    original_results = client.search(
        collection_name=collection_name,
        data=[original_vector],
        limit=30
    )
    all_results.extend([(r, 1.0) for r in original_results])  # 原始查询权重最高
    
    # 2. 同义词扩展
    expanded_queries = expand_with_synonyms(original_query)
    for expanded_query in expanded_queries:
        expanded_vector = get_embedding(expanded_query)
        expanded_results = client.search(
            collection_name=collection_name,
            data=[expanded_vector],
            limit=20
        )
        all_results.extend([(r, 0.8) for r in expanded_results])  # 扩展查询权重稍低
    
    # 3. 意图重写
    rewritten_queries = rewrite_user_intent(original_query)
    for rewritten_query in rewritten_queries:
        rewritten_vector = get_embedding(rewritten_query)
        rewritten_results = client.search(
            collection_name=collection_name,
            data=[rewritten_vector],
            limit=15
        )
        all_results.extend([(r, 0.6) for r in rewritten_results])  # 重写查询权重更低
    
    # 融合结果
    return weighted_merge_results(all_results)

def expand_with_synonyms(query):
    """生成同义词扩展查询"""
    
    # 宠物护理领域的同义词映射
    synonyms_map = {
        "狗": ["犬", "狗狗", "小狗", "宠物狗"],
        "猫": ["猫咪", "小猫", "宠物猫"],
        "洗澡": ["清洁", "清洗", "沐浴"],
        "喂食": ["喂养", "饮食", "进食"],
        "训练": ["训导", "教育", "调教"],
        "生病": ["疾病", "不适", "健康问题"],
        "护理": ["照顾", "保养", "维护"]
    }
    
    expanded = [query]  # 包含原始查询
    
    for word, synonyms in synonyms_map.items():
        if word in query:
            for synonym in synonyms:
                expanded.append(query.replace(word, synonym))
    
    return expanded[:3]  # 限制扩展数量

def rewrite_user_intent(query):
    """根据用户意图重写查询"""
    
    rewritten = []
    
    # 意图模式匹配
    if "怎么" in query or "如何" in query:
        # 添加更具体的操作描述
        rewritten.append(query.replace("怎么", "步骤").replace("如何", "方法"))
        rewritten.append(query + " 教程")
    
    if "什么" in query:
        # 转换为陈述句
        rewritten.append(query.replace("什么", "介绍"))
    
    if "为什么" in query:
        # 添加解释性词语
        rewritten.append(query.replace("为什么", "原因"))
        rewritten.append(query + " 解释")
    
    return rewritten[:2]  # 限制重写数量
```

#### 召回率评估和监控
```python
def evaluate_recall_performance(test_queries, ground_truth_mapping):
    """评估系统召回率性能"""
    
    total_recall = 0
    total_precision = 0
    results_log = []
    
    for query_text, expected_docs in test_queries.items():
        # 执行检索
        query_vector = get_embedding(query_text)
        search_results = client.search(
            collection_name="pet_data_collection",
            data=[query_vector],
            limit=10,
            output_fields=["uuid"]
        )
        
        retrieved_uuids = [r['entity']['uuid'] for r in search_results]
        expected_uuids = set(expected_docs)
        
        # 计算召回率和精确率
        relevant_retrieved = len(set(retrieved_uuids) & expected_uuids)
        recall = relevant_retrieved / len(expected_uuids) if expected_uuids else 0
        precision = relevant_retrieved / len(retrieved_uuids) if retrieved_uuids else 0
        
        total_recall += recall
        total_precision += precision
        
        results_log.append({
            'query': query_text,
            'recall': recall,
            'precision': precision,
            'retrieved_count': len(retrieved_uuids),
            'expected_count': len(expected_uuids)
        })
    
    avg_recall = total_recall / len(test_queries)
    avg_precision = total_precision / len(test_queries)
    
    # 生成评估报告
    print(f"平均召回率: {avg_recall:.3f}")
    print(f"平均精确率: {avg_precision:.3f}")
    print(f"F1分数: {2 * (avg_precision * avg_recall) / (avg_precision + avg_recall):.3f}")
    
    return {
        'average_recall': avg_recall,
        'average_precision': avg_precision,
        'detailed_results': results_log
    }

# 测试用例示例
test_cases = {
    "如何给狗狗洗澡": ["doc_001", "doc_045", "doc_123"],
    "猫咪不吃饭怎么办": ["doc_078", "doc_156", "doc_234"],
    "宠物疫苗接种时间": ["doc_012", "doc_089", "doc_167"],
    "狗狗训练基础知识": ["doc_034", "doc_098", "doc_201"]
}
```

#### 实时召回优化建议
```yaml
召回率优化检查清单:

1. 索引层面:
   - 调整ef参数: 64→128→256逐步测试
   - 增加M参数: 16→32提高图连接密度
   - 考虑使用多个索引: 不同参数的索引服务不同场景

2. 数据质量:
   - 检查chunk切分: 确保语义完整性
   - 验证embedding质量: 使用更好的模型
   - 数据清洗: 去除重复和低质量内容

3. 查询优化:
   - 多路召回: 向量+关键词+语义匹配
   - 查询扩展: 同义词、意图重写
   - 负反馈学习: 根据用户行为调整

4. 系统监控:
   - 定期评估召回率: 每周/月评估
   - A/B测试: 不同策略对比
   - 用户反馈: 收集"没找到"的情况

5. 针对宠物护理场景:
   - 专业词汇扩展: 医学术语、品种名称
   - 多语言支持: 中英文混合查询
   - 图片+文本: 多模态检索(未来)
```

## 3. 数据库设计最佳实践

### 3.1 Collection Schema设计

```python
推荐Schema：
{
    "collection_name": "document_collection",
    "fields": [
        {
            "name": "uuid",
            "type": "VARCHAR",
            "max_length": 64,
            "is_primary": True,
            "description": "文档chunk的唯一标识"
        },
        {
            "name": "document_id", 
            "type": "VARCHAR",
            "max_length": 64,
            "description": "原始文档的ID"
        },
        {
            "name": "chunk_index",
            "type": "INT32",
            "description": "在文档中的chunk序号"
        },
        {
            "name": "title",
            "type": "VARCHAR", 
            "max_length": 1024,
            "description": "文档标题"
        },
        {
            "name": "content",
            "type": "VARCHAR",
            "max_length": 2048,
            "description": "chunk的文本内容"
        },
        {
            "name": "path",
            "type": "VARCHAR",
            "max_length": 256, 
            "description": "文件路径"
        },
        {
            "name": "url",
            "type": "VARCHAR",
            "max_length": 2048,
            "description": "文档URL"
        },
        {
            "name": "metadata",
            "type": "JSON",
            "description": "额外的元数据信息"
        },
        {
            "name": "embedding",
            "type": "FLOAT_VECTOR",
            "dim": 1536,
            "description": "文本的向量表示"
        },
        {
            "name": "created_time",
            "type": "INT64",
            "description": "创建时间戳"
        }
    ],
    "enable_dynamic_field": True
}
```

### 3.2 索引配置

```python
推荐索引配置：
{
    "index_type": "HNSW",  # 高性能近似搜索
    "metric_type": "COSINE",  # 余弦相似度
    "params": {
        "M": 16,  # 连接数
        "efConstruction": 200  # 构建时的搜索范围
    }
}

性能调优：
- M: 16-64（越大精度越高，内存占用越大）
- efConstruction: 100-500（影响构建时间和精度）
- ef: 64-512（搜索时参数，影响召回率和速度）
```

## 4. 性能优化建议

### 4.1 存储优化

```yaml
存储策略：
- 分区策略：按文档类型或时间分区
- 压缩：启用向量压缩节省存储空间
- 内存管理：合理设置collection加载策略
- 备份：定期备份重要数据
```

### 4.2 检索优化

```yaml
检索优化：
- 预过滤：使用标量字段预过滤减少计算量
- 批量查询：多个查询合并处理
- 缓存：对热门查询结果进行缓存
- 异步处理：对非实时查询使用异步处理
```

### 4.3 向量优化

```yaml
向量优化：
- 维度选择：根据模型和精度需求选择合适维度
- 归一化：对向量进行L2归一化提高相似度计算精度
- 量化：使用PQ或其他量化技术减少内存占用
```

## 5. 实际应用建议

### 5.1 宠物护理场景配置

```python
宠物护理RAG推荐配置：
{
    "chunking": {
        "method": "paragraph_with_overlap",
        "max_size": 500,
        "overlap": 50,
        "min_size": 100
    },
    "retrieval": {
        "top_k": 5,
        "min_similarity": 0.4,
        "output_fields": ["uuid", "title", "content", "path", "url"],
        "metric_type": "COSINE"
    },
    "post_processing": {
        "enable_rerank": True,
        "max_final_chunks": 3,
        "similarity_boost_for_title_match": 0.1
    }
}
```

### 5.2 质量监控

```python
质量指标监控：
- 平均检索时间：< 100ms
- 相似度分布：确保高质量结果占比
- 召回率：通过人工评估验证
- 用户满意度：通过反馈收集
```

### 5.3 持续优化

```yaml
优化策略：
1. A/B测试：不同检索策略的效果对比
2. 用户反馈：收集用户对结果质量的评价
3. 数据分析：分析查询模式优化索引
4. 模型升级：定期评估新的embedding模型
5. 参数调优：根据实际使用情况调整参数
```

## 6. 故障排查

### 6.1 常见问题

```yaml
问题排查清单：
- 检索返回空结果：检查collection是否有数据、是否已加载
- 相似度异常：确认embedding模型一致性
- 性能缓慢：检查索引配置和数据量
- 内存占用过高：检查向量维度和数据量
- 结果质量差：检查chunk切分策略和相似度阈值
```

### 6.2 调试工具

```python
调试脚本示例：
def debug_retrieval(collection_name, query_text, top_k=10):
    """调试检索结果"""
    
    # 1. 检查collection状态
    stats = client.get_collection_stats(collection_name)
    print(f"Collection stats: {stats}")
    
    # 2. 执行检索
    results = client.search(...)
    
    # 3. 分析结果
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Similarity: {result['similarity']:.4f}")
        print(f"  Content: {result['entity']['content'][:100]}...")
        print()
```

## 7. 版本兼容性

### 7.1 Dify插件兼容性

```yaml
当前状态：
- Dify版本：1.6.0+
- model-selector：已弃用，使用系统级模型管理
- 推荐方案：使用默认embedding模型，避免插件级配置
```

### 7.2 未来升级路径

```yaml
升级策略：
1. 监控Dify官方插件API变化
2. 保持与最新PyMilvus版本兼容
3. 准备向后兼容性方案
4. 文档和配置的版本管理
```

---

## 总结

本策略指南涵盖了从文档切分到检索优化的完整RAG工作流。重点建议：

1. **切分策略**：使用段落边界+固定长度的混合方案
2. **检索策略**：Top-K + 相似度阈值的智能过滤
3. **性能优化**：合理的索引配置和缓存机制
4. **质量保证**：持续监控和优化

根据实际使用情况，定期评估和调整这些策略，以获得最佳的RAG系统性能。