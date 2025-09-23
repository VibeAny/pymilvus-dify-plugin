# Milvus Plus Vector Database Plugin for Dify

An enhanced Milvus vector database plugin that provides comprehensive vector operations for the Dify platform, including collection management, data operations, text embedding, semantic search, and BM25 keyword search capabilities.

[中文文档](./README_zh.md) | [English](./README.md)

## Acknowledgments

This project is built upon the original Milvus plugin with significant enhancements. We thank the original author for their contribution.

## Features

### 🗂️ Collection Management
- **List Collections**: View all available collections
- **Describe Collection**: Get detailed collection information
- **Collection Stats**: Retrieve collection statistics
- **Check Existence**: Verify if collections exist

### 📥 Data Operations
- **Insert Data**: Add vectors and metadata to collections
- **Upsert Data**: Insert or update existing data
- **Query Data**: Retrieve data by ID or filter conditions
- **Delete Data**: Remove data from collections

### 🔍 Vector Search
- **Similarity Search**: Find similar vectors using various metrics
- **Filtered Search**: Combine vector similarity with metadata filters
- **Multi-Vector Search**: Search with multiple query vectors
- **Custom Parameters**: Adjust search behavior parameters

### ✨ Enhanced Features (New)

#### 🔤 Text Embedding
- **Auto Vectorization**: Convert text to vectors using PyMilvus
- **Multi-Model Support**: Support for OpenAI and Azure OpenAI embedding models
- **Vector Normalization**: Optional L2 vector normalization
- **Dimension Detection**: Automatic vector dimension detection

#### 🔍 Semantic Text Search  
- **End-to-End Search**: Text queries automatically converted to vectors and searched
- **Similarity Filtering**: Support minimum similarity threshold filtering
- **Multiple Distance Metrics**: Support for COSINE, L2, and other distance metrics
- **Flexible Output**: Custom output fields and filter conditions

#### 📝 BM25 Keyword Search
- **Traditional Text Retrieval**: BM25 algorithm-based keyword search
- **Parameter Tuning**: Support for custom k1, b parameter adjustment
- **Fast Response**: Direct text matching without vectorization
- **Hybrid Search**: Can be combined with vector search

## Installation & Configuration

### Connection Configuration
Configure your Milvus connection in the Dify platform:

#### 🔧 Milvus Basic Configuration
- **URI**: Milvus server address (e.g., `http://localhost:19530`)
- **Token**: Authentication token (optional, format: `username:password`)
- **Database**: Target database name (default: `default`)

#### 🤖 Embedding Model Configuration
Choose embedding provider: `openai` or `azure_openai`

##### OpenAI Configuration
- **OpenAI API Key**: Your OpenAI API key
- **OpenAI Base URL**: API base URL (optional)

##### Azure OpenAI Configuration  
- **Azure OpenAI Endpoint**: Azure OpenAI service endpoint
- **Azure OpenAI API Key**: Azure OpenAI API key
- **Azure API Version**: API version (default: 2023-12-01-preview)

## Usage Examples

### Collection Operations
```python
# List all collections
{"operation": "list"}

# Describe collection
{"operation": "describe", "collection_name": "my_collection"}

# Get collection statistics
{"operation": "stats", "collection_name": "my_collection"}

# Check if collection exists
{"operation": "exists", "collection_name": "my_collection"}
```

### Data Operations
```python
# Insert data
{
  "collection_name": "my_collection",
  "data": [{"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": "sample"}]
}

# Vector search
{
  "collection_name": "my_collection",
  "query_vector": [0.1, 0.2, 0.3],
  "limit": 10
}
```

### ✨ Enhanced Features Examples

#### Text Embedding
```python
{
  "text": "This is a text that needs to be vectorized",
  "model": "text-embedding-3-small",
  "normalize": true
}
```

#### Semantic Text Search
```python
{
  "collection_name": "documents",
  "query_text": "artificial intelligence development history",
  "limit": 5,
  "embedding_model": "text-embedding-3-small",
  "metric_type": "COSINE",
  "min_similarity": 0.7,
  "output_fields": "title,content,metadata"
}
```

#### BM25 Keyword Search
```python
{
  "collection_name": "documents", 
  "query_text": "machine learning deep learning",
  "limit": 10,
  "bm25_k1": 1.2,
  "bm25_b": 0.75,
  "output_fields": "title,content,score"
}
```

## Technical Architecture

### Dependencies
- **PyMilvus**: Milvus Python SDK (v2.6.0+)
- **PyMilvus[model]**: Embedding model support
- **requests**: HTTP API calls
- **dify_plugin**: Dify plugin framework

### Tool List
1. **milvus_collection** - Collection management operations
2. **milvus_data** - Data CRUD operations  
3. **milvus_search** - Vector similarity search
4. **milvus_text_embedding** - Text vectorization ✨
5. **milvus_text_search** - Semantic text search ✨
6. **milvus_bm25_search** - BM25 keyword search ✨

### Architecture Features
- **Unified Error Handling**: Consistent error messages and exception handling
- **Connection Pool Management**: Efficient Milvus connection management
- **Dual Client Support**: HTTP API and SDK client coexistence
- **Auto Fallback Mechanism**: Automatic fallback to direct API calls when Azure OpenAI is incompatible

## Development Information

- **Version**: 0.1.3
- **Author**: VibeAny (Original) + Enhanced by ZeroZ Lab
- **License**: MIT License
- **Minimum Dify Version**: 1.5.0

## Changelog

### v0.1.3 (Enhanced)
- ✨ Added text embedding tool
- ✨ Added semantic text search tool  
- ✨ Added BM25 keyword search tool
- 🔧 Refactored to unified tool architecture
- 🔧 Enhanced error handling and logging
- 🔧 Optimized user configuration interface
- 🔧 Support for both OpenAI and Azure OpenAI providers



