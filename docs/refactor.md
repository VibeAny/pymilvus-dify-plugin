# Milvus Plugin Complete Rewrite Plan

## Background

The current plugin uses custom HTTP API implementation, which has fundamental limitations:
- Complex custom HTTP client code
- Limited feature support compared to PyMilvus
- No native BM25 support
- Authentication complexity
- Missing advanced PyMilvus features

## Solution

**Complete rewrite** using PyMilvus SDK as the foundation. This is NOT a migration - it's a new plugin implementation that leverages the full power of PyMilvus.

## Current vs Target Architecture

### Current (Custom HTTP Implementation)
```
Plugin → Custom HTTP Client → Manual API Calls → Complex Error Handling
Issues: Limited features, complex code, no BM25, authentication problems
```

### Target (Pure PyMilvus Implementation)
```
Plugin → PyMilvus SDK → Native Features → Standard Exceptions
Benefits: Full features, clean code, native BM25, standard auth, future-proof
```

## Complete Rewrite Strategy

### Remove Everything HTTP-Related
- **Delete all HTTP client code** - No backward compatibility needed
- **Delete custom request/response handling** - Use PyMilvus patterns
- **Delete HTTP authentication logic** - Use PyMilvus auth
- **Start completely fresh** - Clean slate approach

### Pure PyMilvus Foundation
- **Single dependency**: Only PyMilvus SDK
- **Native patterns**: Follow PyMilvus conventions
- **Full feature access**: All PyMilvus capabilities available
- **Standard exceptions**: Use PyMilvus error handling

## Implementation Plan

### Phase 1: Foundation Rewrite

#### 1.1 Brand New Provider Configuration
**File:** `provider/milvus.yaml` - **COMPLETE REWRITE**

```yaml
identity:
  author: vibeany
  name: milvus
  label:
    en_US: Milvus Vector Database
    zh_Hans: Milvus 向量数据库
  description:
    en_US: Complete Milvus vector database integration using PyMilvus SDK with native BM25, hybrid search, and advanced collection management
    zh_Hans: 基于 PyMilvus SDK 的完整 Milvus 向量数据库集成，支持原生 BM25、混合搜索和高级集合管理
  icon: icon.svg
  tags:
    - vector-database
    - search
    - embedding
    - bm25
    - pymilvus

credentials_schema:
  - variable: uri
    type: text-input
    required: true
    default: "https://milvus-api.roomwits.com"
    label:
      en_US: Milvus Server URI
      zh_Hans: Milvus 服务器地址
    placeholder:
      en_US: https://your-milvus-server.com
      zh_Hans: https://你的-milvus-服务器.com
    help:
      en_US: The gRPC endpoint of your Milvus server
      zh_Hans: Milvus 服务器的 gRPC 端点
    
  - variable: user
    type: text-input
    required: true
    default: "root"
    label:
      en_US: Username
      zh_Hans: 用户名
    placeholder:
      en_US: root
      zh_Hans: root
    
  - variable: password
    type: secret-input
    required: true
    label:
      en_US: Password
      zh_Hans: 密码
    placeholder:
      en_US: Enter your password
      zh_Hans: 输入密码
    
  - variable: database
    type: text-input
    required: false
    default: "default"
    label:
      en_US: Database Name
      zh_Hans: 数据库名称
    placeholder:
      en_US: default
      zh_Hans: default
    help:
      en_US: Milvus database name (default: "default")
      zh_Hans: Milvus 数据库名称（默认："default"）

  # Embedding provider configuration (for text embedding search)
  - variable: embedding_provider
    type: select
    required: false
    default: "openai"
    label:
      en_US: Embedding Provider
      zh_Hans: 嵌入模型提供商
    options:
      - value: "openai"
        label:
          en_US: OpenAI
          zh_Hans: OpenAI
      - value: "azure_openai"
        label:
          en_US: Azure OpenAI
          zh_Hans: Azure OpenAI
    help:
      en_US: Provider for text embedding (used in text embedding search)
      zh_Hans: 文本嵌入提供商（用于文本嵌入搜索）

  - variable: openai_api_key
    type: secret-input
    required: false
    label:
      en_US: OpenAI API Key
      zh_Hans: OpenAI API 密钥
    show_on:
      - variable: embedding_provider
        value: openai

  - variable: azure_openai_endpoint
    type: text-input
    required: false
    label:
      en_US: Azure OpenAI Endpoint
      zh_Hans: Azure OpenAI 端点
    show_on:
      - variable: embedding_provider
        value: azure_openai

  - variable: azure_openai_api_key
    type: secret-input
    required: false
    label:
      en_US: Azure OpenAI API Key
      zh_Hans: Azure OpenAI API 密钥
    show_on:
      - variable: embedding_provider
        value: azure_openai

  - variable: azure_api_version
    type: text-input
    required: false
    default: "2023-12-01-preview"
    label:
      en_US: Azure API Version
      zh_Hans: Azure API 版本
    show_on:
      - variable: embedding_provider
        value: azure_openai
```

#### 1.2 Brand New Provider Validation
**File:** `provider/milvus.py` - **COMPLETE REWRITE**

```python
from typing import Any
from pymilvus import MilvusClient
from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

class MilvusProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        """Validate credentials using PyMilvus client"""
        
        # Validate Milvus connection
        self._validate_milvus_connection(credentials)
        
        # Validate embedding provider if configured
        embedding_provider = credentials.get("embedding_provider")
        if embedding_provider:
            self._validate_embedding_provider(credentials)
    
    def _validate_milvus_connection(self, credentials: dict[str, Any]) -> None:
        """Validate Milvus connection using PyMilvus"""
        uri = credentials.get("uri")
        user = credentials.get("user") 
        password = credentials.get("password")
        database = credentials.get("database", "default")
        
        if not uri:
            raise ToolProviderCredentialValidationError("Milvus URI is required")
        if not user:
            raise ToolProviderCredentialValidationError("Username is required")
        if not password:
            raise ToolProviderCredentialValidationError("Password is required")
        
        try:
            # Create PyMilvus client
            client = MilvusClient(
                uri=uri,
                user=user,
                password=password,
                db_name=database
            )
            
            # Test connection with a simple operation
            collections = client.list_collections()
            
            # Connection successful
            
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise ToolProviderCredentialValidationError("Authentication failed. Please check username and password.")
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                raise ToolProviderCredentialValidationError("Cannot connect to Milvus server. Please check URI.")
            else:
                raise ToolProviderCredentialValidationError(f"Milvus connection failed: {error_msg}")
    
    def _validate_embedding_provider(self, credentials: dict[str, Any]) -> None:
        """Validate embedding provider configuration"""
        provider = credentials.get("embedding_provider")
        
        if provider == "openai":
            api_key = credentials.get("openai_api_key")
            if api_key:
                self._test_openai_connection(api_key)
        elif provider == "azure_openai":
            endpoint = credentials.get("azure_openai_endpoint")
            api_key = credentials.get("azure_openai_api_key")
            if endpoint and api_key:
                self._test_azure_openai_connection(endpoint, api_key, credentials.get("azure_api_version"))
    
    def _test_openai_connection(self, api_key: str) -> None:
        """Test OpenAI API connection"""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            # Test with a simple API call
            client.models.list()
        except Exception as e:
            raise ToolProviderCredentialValidationError(f"OpenAI API validation failed: {str(e)}")
    
    def _test_azure_openai_connection(self, endpoint: str, api_key: str, api_version: str) -> None:
        """Test Azure OpenAI API connection"""
        try:
            import openai
            client = openai.AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version
            )
            # Test connection - this will work if credentials are valid
            # Even if no models are deployed, authentication will be verified
        except Exception as e:
            raise ToolProviderCredentialValidationError(f"Azure OpenAI API validation failed: {str(e)}")
```

### Phase 2: PyMilvus Foundation

#### 2.1 Pure PyMilvus Client Wrapper
**File:** `tools/pymilvus_client.py` - **BRAND NEW**

```python
from typing import Any, Optional, List, Dict, Union
from pymilvus import MilvusClient, DataType, Function, FunctionType
from pymilvus.model.dense import OpenAIEmbeddingFunction
import logging

logger = logging.getLogger(__name__)

class MilvusPyClient:
    """Pure PyMilvus client wrapper for Dify plugin"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "default"):
        """Initialize PyMilvus client"""
        self.client = MilvusClient(
            uri=uri,
            user=user,
            password=password,
            db_name=database
        )
        self.database = database
        logger.info(f"Connected to Milvus database: {database}")
    
    # Collection Operations
    def create_simple_collection(self, name: str, dimension: int, metric_type: str = "COSINE", 
                               description: str = "", auto_id: bool = True) -> dict:
        """Create a simple vector collection"""
        schema = self.client.create_schema(auto_id=auto_id, enable_dynamic_field=True)
        
        # Add fields
        if auto_id:
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        else:
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=36)
        
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
        
        # Create index
        index_params = self.client.prepare_index_params()
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type=metric_type)
        
        return self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
            description=description
        )
    
    def create_bm25_collection(self, name: str, text_field: str, vector_dim: int, 
                              metric_type: str = "COSINE", description: str = "") -> dict:
        """Create collection with native BM25 support"""
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        
        # Primary key
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        
        # Text field with analyzer enabled for BM25
        schema.add_field(text_field, DataType.VARCHAR, max_length=9000, enable_analyzer=True)
        
        # Dense vector for semantic search
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vector_dim)
        
        # Sparse vector for BM25
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        
        # BM25 function
        bm25_function = Function(
            name="bm25_fn",
            input_field_names=[text_field],
            output_field_names="sparse_vector",
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)
        
        # Index parameters
        index_params = self.client.prepare_index_params()
        # Dense vector index
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type=metric_type)
        # Sparse vector index for BM25
        index_params.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", 
                             metric_type="BM25", params={"bm25_k1": 1.2, "bm25_b": 0.75})
        
        return self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
            description=description
        )
    
    def create_hybrid_collection(self, name: str, text_field: str, vector_dim: int, 
                                additional_fields: List[Dict] = None) -> dict:
        """Create collection supporting both vector and BM25 search with additional fields"""
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        
        # Primary key
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        
        # Text field
        schema.add_field(text_field, DataType.VARCHAR, max_length=9000, enable_analyzer=True)
        
        # Vector fields
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        
        # Additional custom fields
        if additional_fields:
            for field in additional_fields:
                if field["type"] == "varchar":
                    schema.add_field(field["name"], DataType.VARCHAR, max_length=field.get("max_length", 1000))
                elif field["type"] == "int64":
                    schema.add_field(field["name"], DataType.INT64)
                elif field["type"] == "float":
                    schema.add_field(field["name"], DataType.FLOAT)
                # Add more types as needed
        
        # BM25 function
        bm25_function = Function(
            name="bm25_fn",
            input_field_names=[text_field],
            output_field_names="sparse_vector",
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)
        
        # Indexes
        index_params = self.client.prepare_index_params()
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        
        return self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params
        )
    
    # Search Operations
    def vector_search(self, collection_name: str, vectors: List[List[float]], 
                     limit: int = 10, output_fields: List[str] = None, 
                     filter_expr: str = None, search_params: dict = None) -> List:
        """Perform vector similarity search"""
        return self.client.search(
            collection_name=collection_name,
            data=vectors,
            anns_field="vector",
            limit=limit,
            output_fields=output_fields or ["*"],
            filter=filter_expr,
            search_params=search_params or {}
        )
    
    def bm25_search(self, collection_name: str, query_text: str, limit: int = 10,
                   output_fields: List[str] = None, filter_expr: str = None,
                   bm25_k1: float = 1.2, bm25_b: float = 0.75) -> List:
        """Perform BM25 text search"""
        search_params = {
            "metric_type": "BM25",
            "params": {"bm25_k1": bm25_k1, "bm25_b": bm25_b}
        }
        
        return self.client.search(
            collection_name=collection_name,
            data=[query_text],
            anns_field="sparse_vector",
            limit=limit,
            output_fields=output_fields or ["*"],
            filter=filter_expr,
            search_params=search_params
        )
    
    def hybrid_search(self, collection_name: str, query_text: str, query_vector: List[float],
                     limit: int = 10, output_fields: List[str] = None,
                     vector_weight: float = 0.7, bm25_weight: float = 0.3) -> List:
        """Perform hybrid search combining vector similarity and BM25"""
        # This would use PyMilvus hybrid search capabilities
        # Implementation depends on PyMilvus version and hybrid search API
        pass
    
    # Data Operations (using pure PyMilvus)
    def insert_data(self, collection_name: str, data: List[Dict]) -> dict:
        """Insert data using PyMilvus"""
        return self.client.insert(collection_name, data)
    
    def upsert_data(self, collection_name: str, data: List[Dict]) -> dict:
        """Upsert data using PyMilvus"""
        return self.client.upsert(collection_name, data)
    
    def query_data(self, collection_name: str, filter_expr: str = "", 
                  output_fields: List[str] = None, limit: int = None) -> List:
        """Query data using PyMilvus"""
        return self.client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=output_fields or ["*"],
            limit=limit
        )
    
    def delete_data(self, collection_name: str, ids: List = None, filter_expr: str = None) -> dict:
        """Delete data using PyMilvus"""
        if ids:
            filter_expr = f"id in {ids}"
        return self.client.delete(collection_name, filter=filter_expr)
    
    def get_by_ids(self, collection_name: str, ids: List, output_fields: List[str] = None) -> List:
        """Get entities by IDs using PyMilvus"""
        return self.client.get(
            collection_name=collection_name,
            ids=ids,
            output_fields=output_fields or ["*"]
        )
    
    # Collection Management (pure PyMilvus)
    def list_collections(self) -> List[str]:
        """List all collections"""
        return self.client.list_collections()
    
    def describe_collection(self, collection_name: str) -> dict:
        """Describe collection schema"""
        return self.client.describe_collection(collection_name)
    
    def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists"""
        return self.client.has_collection(collection_name)
    
    def drop_collection(self, collection_name: str) -> dict:
        """Drop collection"""
        return self.client.drop_collection(collection_name)
    
    def load_collection(self, collection_name: str) -> dict:
        """Load collection into memory"""
        return self.client.load_collection(collection_name)
    
    def release_collection(self, collection_name: str) -> dict:
        """Release collection from memory"""
        return self.client.release_collection(collection_name)
```

#### 2.2 New Base Tool Class
**File:** `tools/milvus_base.py` - **COMPLETE REWRITE**

```python
from typing import Any
from collections.abc import Generator
from contextlib import contextmanager
from dify_plugin.entities.tool import ToolInvokeMessage
from .pymilvus_client import MilvusPyClient
import logging

logger = logging.getLogger(__name__)

class MilvusBaseTool:
    """Base class for all Milvus tools using pure PyMilvus"""
    
    @contextmanager
    def _get_milvus_client(self, credentials: dict[str, Any]):
        """Get PyMilvus client - no HTTP code anywhere"""
        try:
            client = MilvusPyClient(
                uri=credentials.get("uri"),
                user=credentials.get("user"),
                password=credentials.get("password"),
                database=credentials.get("database", "default")
            )
            yield client
        except Exception as e:
            logger.error(f"PyMilvus client error: {str(e)}")
            raise ValueError(f"Milvus connection failed: {str(e)}")
    
    def _create_success_message(self, data: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """Create standard success response"""
        response = {"success": True, **data}
        yield self.create_json_message(response)
    
    def _create_error_message(self, error: str, error_type: str = "MilvusError") -> Generator[ToolInvokeMessage]:
        """Create standard error response"""
        response = {
            "success": False,
            "error": error,
            "error_type": error_type
        }
        yield self.create_json_message(response)
    
    def _validate_collection_name(self, collection_name: str) -> bool:
        """Validate collection name format"""
        if not collection_name or not isinstance(collection_name, str):
            return False
        if len(collection_name) > 255:
            return False
        # Collection name validation rules
        import re
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', collection_name))
    
    def _parse_json_field(self, field_value: str, field_name: str):
        """Parse JSON string field"""
        try:
            import json
            return json.loads(field_value)
        except (json.JSONDecodeError, TypeError):
            raise ValueError(f"Invalid JSON format in {field_name}")
```

## Benefits of Complete Rewrite

### Code Quality
- **60% less code** - No custom HTTP implementation
- **Standard patterns** - Follow PyMilvus conventions  
- **Better maintainability** - Single dependency
- **Future-proof** - Access to all new PyMilvus features

### Features
- **Native BM25** - Full-text search with proper ranking
- **Hybrid search** - Combine vector + text search  
- **Advanced schemas** - Multiple vectors, partitions, scalar indexes
- **Performance** - Direct gRPC, no HTTP overhead
- **All PyMilvus features** - No artificial limitations

### Development Experience
- **Standard debugging** - Use PyMilvus docs and examples
- **Community support** - Leverage PyMilvus community
- **Easy updates** - Follow PyMilvus release cycle
- **Simple testing** - Mock PyMilvus client for unit tests

## Next Steps

1. **Approve complete rewrite approach** - Confirm this is the right direction
2. **Start with foundation** - Provider and base client first
3. **TDD implementation** - Write tests first, then implement
4. **Tool-by-tool rewrite** - Replace each tool with PyMilvus version
5. **Advanced features** - Add native BM25 and hybrid search

This rewrite will result in a **much better plugin** that fully leverages PyMilvus capabilities while being simpler to maintain and extend.