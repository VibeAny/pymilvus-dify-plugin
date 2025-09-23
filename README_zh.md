# Milvus Plus Vector Database 插件 (Dify)

一个增强版的 Milvus 向量数据库插件，为 Dify 平台提供完整的向量操作功能，包括集合管理、数据操作、文本嵌入、语义搜索和 BM25 关键词搜索。

[中文文档](./README_zh.md) | [English](./README.md)

## 致谢

本项目基于原始 Milvus 插件进行增强开发，感谢原作者的贡献。

## 功能特性

### 🗂️ 集合管理
- **列出集合**: 查看所有可用集合
- **描述集合**: 获取集合的详细信息
- **集合统计**: 检索集合统计数据
- **检查存在性**: 验证集合是否存在

### 📥 数据操作
- **插入数据**: 向集合中添加向量和元数据
- **更新插入数据**: 插入或更新现有数据
- **查询数据**: 通过ID或过滤条件检索数据
- **删除数据**: 从集合中移除数据

### 🔍 向量搜索
- **相似度搜索**: 使用各种指标查找相似向量
- **过滤搜索**: 结合向量相似度和元数据过滤
- **多向量搜索**: 使用多个查询向量进行搜索
- **自定义参数**: 调整搜索行为参数

### ✨ 新增功能 (Enhanced Features)

#### 🔤 文本嵌入 (Text Embedding)
- **自动向量化**: 使用 PyMilvus 将文本转换为向量
- **多模型支持**: 支持 OpenAI 和 Azure OpenAI 嵌入模型
- **向量标准化**: 可选的 L2 向量标准化
- **维度检测**: 自动检测向量维度

#### 🔍 智能文本搜索 (Semantic Text Search)  
- **端到端搜索**: 文本查询自动转换为向量并执行搜索
- **相似度过滤**: 支持最小相似度阈值过滤
- **多种距离度量**: 支持 COSINE、L2 等距离度量
- **灵活输出**: 自定义输出字段和过滤条件

#### 📝 BM25 关键词搜索 (BM25 Keyword Search)
- **传统文本检索**: 基于 BM25 算法的关键词搜索
- **参数调优**: 支持 k1、b 参数自定义调节
- **快速响应**: 无需向量化的直接文本匹配
- **混合搜索**: 可与向量搜索结合使用

## 安装与配置

### 连接配置
在 Dify 平台中配置您的 Milvus 连接:

#### 🔧 Milvus 基础配置
- **URI**: Milvus 服务器地址 (例如 `http://localhost:19530`)
- **Token**: 认证令牌 (可选，格式: `username:password`)
- **Database**: 目标数据库名称 (默认: `default`)

#### 🤖 嵌入模型配置
选择嵌入提供商：`openai` 或 `azure_openai`

##### OpenAI 配置
- **OpenAI API Key**: 您的 OpenAI API 密钥
- **OpenAI Base URL**: API 基础 URL (可选)

##### Azure OpenAI 配置  
- **Azure OpenAI Endpoint**: Azure OpenAI 服务端点
- **Azure OpenAI API Key**: Azure OpenAI API 密钥
- **Azure API Version**: API 版本 (默认: 2023-12-01-preview)

## 使用示例

### 集合操作
```python
# 列出所有集合
{"operation": "list"}

# 描述集合
{"operation": "describe", "collection_name": "my_collection"}

# 获取集合统计
{"operation": "stats", "collection_name": "my_collection"}

# 检查集合是否存在
{"operation": "exists", "collection_name": "my_collection"}
```

### 数据操作
```python
# 插入数据
{
  "collection_name": "my_collection",
  "data": [{"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": "sample"}]
}

# 向量搜索
{
  "collection_name": "my_collection",
  "query_vector": [0.1, 0.2, 0.3],
  "limit": 10
}
```

### ✨ 新功能使用示例

#### 文本嵌入
```python
{
  "text": "这是一段需要向量化的文本",
  "model": "text-embedding-3-small",
  "normalize": true
}
```

#### 智能文本搜索
```python
{
  "collection_name": "documents",
  "query_text": "人工智能的发展历史",
  "limit": 5,
  "embedding_model": "text-embedding-3-small",
  "metric_type": "COSINE",
  "min_similarity": 0.7,
  "output_fields": "title,content,metadata"
}
```

#### BM25 关键词搜索
```python
{
  "collection_name": "documents", 
  "query_text": "机器学习 深度学习",
  "limit": 10,
  "bm25_k1": 1.2,
  "bm25_b": 0.75,
  "output_fields": "title,content,score"
}
```

## 技术架构

### 依赖库
- **PyMilvus**: Milvus Python SDK (v2.6.0+)
- **PyMilvus[model]**: 嵌入模型支持
- **requests**: HTTP API 调用
- **dify_plugin**: Dify 插件框架

### 工具列表
1. **milvus_collection** - 集合管理操作
2. **milvus_data** - 数据增删改查操作  
3. **milvus_search** - 向量相似度搜索
4. **milvus_text_embedding** - 文本向量化 ✨
5. **milvus_text_search** - 智能文本搜索 ✨
6. **milvus_bm25_search** - BM25 关键词搜索 ✨

### 架构特点
- **统一错误处理**: 一致的错误信息和异常处理
- **连接池管理**: 高效的 Milvus 连接管理
- **双客户端支持**: HTTP API 和 SDK 客户端并存
- **自动回退机制**: Azure OpenAI 不兼容时自动回退到直接 API 调用

## 开发信息

- **版本**: 0.1.3
- **作者**: VibeAny (原作者) + Enhanced by ZeroZ Lab
- **许可证**: MIT 许可证
- **最低 Dify 版本**: 1.5.0

## 更新日志

### v0.1.3 (Enhanced)
- ✨ 新增文本嵌入工具
- ✨ 新增智能文本搜索工具  
- ✨ 新增 BM25 关键词搜索工具
- 🔧 重构为统一的工具架构
- 🔧 增强错误处理和日志记录
- 🔧 优化用户配置界面
- 🔧 支持 OpenAI 和 Azure OpenAI 双提供商 