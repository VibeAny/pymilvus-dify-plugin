# Milvus Vector Database Plugin for Dify

A comprehensive Milvus vector database plugin for the Dify platform, providing complete vector database operations including collection management, data operations, vector search, text embedding, and native BM25 search capabilities.

## Features

### 🗂️ Collection Management
- **Create Collections**: Create new vector collections with custom schemas
- **List Collections**: View all available collections in your database
- **Describe Collections**: Get detailed schema and configuration information
- **Drop Collections**: Remove collections with safety confirmations

### 📥 Data Operations
- **Insert Data**: Add vectors and metadata to collections
- **Query Data**: Retrieve data using filters and conditions
- **Delete Data**: Remove specific records using filter expressions
- **Bulk Operations**: Efficient batch processing for large datasets

### 🔍 Vector Search
- **Similarity Search**: Find similar vectors using COSINE, L2, or IP metrics
- **Advanced Search**: Support for custom search parameters and filters
- **Multi-field Output**: Retrieve specific fields from search results
- **Performance Tuning**: Configurable search parameters for optimal performance

### ✨ Text Processing & Search
- **Text Embedding**: Convert text to vectors using OpenAI or Azure OpenAI
- **Semantic Search**: End-to-end text-to-vector search workflow
- **BM25 Search**: Traditional keyword-based search with BM25 algorithm
- **Hybrid Search**: Combine vector and keyword search for best results

### 🎯 AI-Focused Capabilities
- **Multiple Embedding Providers**: OpenAI and Azure OpenAI integration
- **Fallback Mechanisms**: Robust handling when embedding services are unavailable
- **Schema Flexibility**: Support for various vector dimensions and data types
- **BM25 Collections**: Native support for text analysis and BM25 indexing

## Installation & Setup

### Prerequisites
- Milvus 2.3.0+ server instance
- Valid OpenAI API key (for text embedding features)
- Dify platform 1.5.0+

### Configuration

#### 1. Milvus Database Connection
Configure your Milvus connection parameters:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `uri` | string | ✅ | Milvus server URI (e.g., `https://your-milvus.com:443`) |
| `user` | string | ✅ | Username for authentication |
| `password` | string | ✅ | Password for authentication |
| `database` | string | ❌ | Database name (default: `default`) |

#### 2. Embedding Provider Configuration
Choose your text embedding provider:

**OpenAI Configuration:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `embedding_provider` | select | ❌ | Set to `openai` |
| `openai_api_key` | string | ✅ | Your OpenAI API key |
| `openai_base_url` | string | ❌ | Custom API base URL |

**Azure OpenAI Configuration:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `embedding_provider` | select | ❌ | Set to `azure_openai` |
| `azure_openai_api_key` | string | ✅ | Azure OpenAI API key |
| `azure_openai_endpoint` | string | ✅ | Azure OpenAI service endpoint |
| `azure_api_version` | string | ❌ | API version (default: `2023-12-01-preview`) |

## Available Tools

### 1. Collection Management

#### Create Collection
Create a new vector collection with custom schema:
```
Parameters:
- collection_name: Name for the new collection
- dimension: Vector dimension (1-32768)
- metric_type: Distance metric (COSINE, L2, IP)
- description: Optional collection description
- enable_bm25: Enable BM25 search support (boolean)
```

#### List Collections
List all collections in the database:
```
No parameters required
```

#### Describe Collection
Get detailed information about a collection:
```
Parameters:
- collection_name: Name of the collection to describe
```

#### Drop Collection
Delete a collection permanently:
```
Parameters:
- collection_name: Name of the collection to delete
- confirm_delete: Confirmation flag (must be true)
```

### 2. Data Operations

#### Insert Data
Add data to a collection:
```
Parameters:
- collection_name: Target collection name
- data: JSON array of records to insert
```

Example data format:
```json
[
  {
    \"id\": 1,
    \"vector\": [0.1, 0.2, 0.3, ...],
    \"text\": \"Sample document content\",
    \"metadata\": {\"category\": \"example\"}
  }
]
```

#### Query Data
Retrieve data using filter conditions:
```
Parameters:
- collection_name: Target collection name
- filter: Filter expression (e.g., \"id in [1, 2, 3]\")
- output_fields: Fields to return (comma-separated)
- limit: Maximum number of results
- offset: Number of results to skip
```

#### Delete Data
Remove data using filter conditions:
```
Parameters:
- collection_name: Target collection name
- filter: Filter expression to select records
- confirm_delete: Confirmation flag (must be true)
```

### 3. Search Operations

#### Vector Search
Find similar vectors:
```
Parameters:
- collection_name: Target collection name
- query_vector: Query vector as JSON array
- limit: Number of results to return
- output_fields: Fields to include in results
- filter: Optional filter expression
- metric_type: Distance metric (COSINE, L2, IP)
```

#### Text Embedding
Convert text to vectors:
```
Parameters:
- text: Text to convert to vector
- model_name: Embedding model name (e.g., \"text-embedding-3-small\")
```

#### Text Search
Search using natural language queries:
```
Parameters:
- collection_name: Target collection name
- query_text: Natural language query
- model_name: Embedding model name
- limit: Number of results to return
- output_fields: Fields to include in results
- similarity_threshold: Minimum similarity score
```

#### BM25 Search
Keyword-based text search:
```
Parameters:
- collection_name: Target collection name
- query_text: Search keywords
- limit: Number of results to return
- k1: BM25 k1 parameter (default: 1.2)
- b: BM25 b parameter (default: 0.75)
```

## Usage Examples

### Basic Collection Workflow
1. **Create a collection** with vector and text fields
2. **Insert data** with vectors and text content
3. **Search** using vectors or natural language
4. **Query** specific records using filters

### Text-Based AI Workflow
1. **Create a BM25-enabled collection** for text search
2. **Insert documents** with text content
3. **Use text search** for semantic similarity
4. **Use BM25 search** for keyword matching
5. **Combine results** for hybrid search

### Advanced Use Cases
- **Document retrieval systems** with semantic search
- **Knowledge base search** with BM25 + vector search
- **Content recommendation** using vector similarity
- **Multi-modal search** combining text and metadata filters

## Technical Architecture

### PyMilvus Integration
- Built on PyMilvus 2.6.0+ with gRPC connectivity
- Native support for all Milvus features
- Efficient connection management and error handling

### Text Processing Pipeline
- Direct integration with OpenAI/Azure OpenAI APIs
- Fallback mechanisms for embedding service issues
- Support for multiple embedding models and dimensions

### BM25 Implementation
- Native Milvus BM25 functions for optimal performance
- Sparse vector indexing for fast text retrieval
- Configurable ranking parameters

### Error Handling & Reliability
- Comprehensive input validation
- User-friendly error messages
- Automatic retry mechanisms for transient failures
- Safe deletion confirmations for destructive operations

## Performance & Best Practices

### Collection Design
- Choose appropriate vector dimensions for your use case
- Use meaningful field names and proper data types
- Enable BM25 only when text search is needed

### Search Optimization
- Use filters to narrow search space
- Adjust search parameters based on your data characteristics
- Consider hybrid search for comprehensive results

### Data Management
- Use batch operations for large datasets
- Implement proper error handling in your applications
- Regular maintenance and monitoring of collection performance

## Security & Privacy

### Data Protection
- All data is processed according to your Milvus server configuration
- Text sent to embedding providers follows their respective privacy policies
- No user data is stored within the plugin itself

### Authentication
- Secure credential storage within Dify
- Support for user/password authentication with Milvus
- API key management for embedding providers

## Support & Maintenance

For technical support and feature requests, please contact the plugin maintainer through the provided support channels.

### System Requirements
- Milvus 2.3.0 or higher
- Python 3.8+ runtime environment  
- Dify platform 1.5.0 or higher

### Plugin Information
- **Version**: 1.0.0
- **License**: MIT
- **Author**: AI Plugin Developer
- **Maintenance**: Active development and support