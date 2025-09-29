# Test-Driven Development (TDD) Strategy for Milvus Plugin Refactoring

## TDD Overview

We will follow **Red-Green-Refactor** cycle for all plugin components:
1. **Red:** Write failing test first
2. **Green:** Write minimal code to make test pass
3. **Refactor:** Improve code while keeping tests green

## Testing Strategy

### Testing Pyramid

```
    /\
   /  \ E2E Tests (Few)
  /____\ Integration Tests (Some)  
 /______\ Unit Tests (Many)
```

- **70% Unit Tests** - Fast, isolated, mock dependencies
- **20% Integration Tests** - Real components, test interactions
- **10% E2E Tests** - Full workflows, real Milvus instance

## Test Structure

### Directory Structure
```
tests/
├── unit/
│   ├── test_provider.py
│   ├── test_base_client.py
│   ├── test_collection_tools.py
│   ├── test_data_tools.py
│   ├── test_search_tools.py
│   └── test_bm25.py
├── integration/
│   ├── test_milvus_connection.py
│   ├── test_collection_workflows.py
│   ├── test_data_workflows.py
│   └── test_search_workflows.py
├── e2e/
│   └── test_complete_workflows.py
├── fixtures/
│   ├── milvus_mock.py
│   ├── test_data.py
│   └── test_collections.py
└── conftest.py
```

## Phase-by-Phase TDD Implementation

### Phase 1: Provider & Configuration TDD

#### 1.1 Provider Validation Tests

**File:** `tests/unit/test_provider.py`

```python
# Red Phase - Write failing tests first
import pytest
from unittest.mock import patch, MagicMock
from provider.milvus import MilvusProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

class TestMilvusProvider:
    
    def test_validate_credentials_success(self):
        """Test successful credential validation"""
        provider = MilvusProvider()
        credentials = {
            "uri": "https://milvus-api.roomwits.com",
            "user": "root",
            "password": "test_password",
            "database": "default"
        }
        # Should not raise exception
        provider._validate_credentials(credentials)
    
    def test_validate_credentials_missing_uri(self):
        """Test validation fails with missing URI"""
        provider = MilvusProvider()
        credentials = {"user": "root", "password": "test"}
        with pytest.raises(ToolProviderCredentialValidationError):
            provider._validate_credentials(credentials)
    
    @patch('pymilvus.MilvusClient')
    def test_milvus_connection_success(self, mock_client):
        """Test successful Milvus connection"""
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.list_collections.return_value = []
        
        provider = MilvusProvider()
        credentials = {
            "uri": "https://milvus-api.roomwits.com",
            "user": "root", 
            "password": "test"
        }
        # Should not raise exception
        provider._validate_milvus_connection(credentials)
    
    @patch('pymilvus.MilvusClient')
    def test_milvus_connection_auth_failure(self, mock_client):
        """Test Milvus connection auth failure"""
        mock_client.side_effect = Exception("Authentication failed")
        
        provider = MilvusProvider()
        credentials = {
            "uri": "https://milvus-api.roomwits.com",
            "user": "wrong_user",
            "password": "wrong_password"
        }
        with pytest.raises(ToolProviderCredentialValidationError):
            provider._validate_milvus_connection(credentials)
```

**TDD Process:**
1. **Red:** Write tests → All fail (provider not updated yet)
2. **Green:** Update `provider/milvus.py` → Tests pass
3. **Refactor:** Improve implementation → Tests stay green

#### 1.2 Configuration Schema Tests

**File:** `tests/unit/test_provider_schema.py`

```python
import yaml
from pathlib import Path

class TestProviderSchema:
    
    def test_provider_yaml_structure(self):
        """Test provider.yaml has correct structure"""
        provider_path = Path("provider/milvus.yaml")
        with open(provider_path) as f:
            config = yaml.safe_load(f)
        
        # Test required fields exist
        assert "credentials_schema" in config
        credentials = config["credentials_schema"]
        
        # Test user field
        user_field = next(f for f in credentials if f["variable"] == "user")
        assert user_field["type"] == "text-input"
        assert user_field["default"] == "root"
        
        # Test password field  
        password_field = next(f for f in credentials if f["variable"] == "password")
        assert password_field["type"] == "secret-input"
        assert password_field["required"] == True
```

### Phase 2: Base Client TDD

#### 2.1 PyMilvus Wrapper Tests

**File:** `tests/unit/test_base_client.py`

```python
import pytest
from unittest.mock import patch, MagicMock
from tools.milvus_pymilvus_client import MilvusClientWrapper

class TestMilvusClientWrapper:
    
    @patch('pymilvus.MilvusClient')
    def test_client_initialization(self, mock_client):
        """Test client wrapper initialization"""
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        wrapper = MilvusClientWrapper(
            uri="https://milvus-api.roomwits.com",
            user="root",
            password="test",
            database="default"
        )
        
        assert wrapper.client == mock_instance
        mock_client.assert_called_once_with(
            uri="https://milvus-api.roomwits.com",
            user="root",
            password="test",
            db_name="default"
        )
    
    @patch('pymilvus.MilvusClient')
    def test_list_collections(self, mock_client):
        """Test list collections functionality"""
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.list_collections.return_value = ["collection1", "collection2"]
        
        wrapper = MilvusClientWrapper("uri", "user", "pass", "db")
        collections = wrapper.list_collections()
        
        assert collections == ["collection1", "collection2"]
        mock_instance.list_collections.assert_called_once()
    
    def test_error_handling(self):
        """Test error handling and logging"""
        # Test connection failures, timeouts, etc.
        pass
```

#### 2.2 Base Tool Class Tests

**File:** `tests/unit/test_milvus_base.py`

```python
from tools.milvus_base import MilvusBaseTool
from unittest.mock import patch, MagicMock

class TestMilvusBaseTool:
    
    def test_get_milvus_client_context(self):
        """Test PyMilvus client context manager"""
        tool = MilvusBaseTool()
        credentials = {
            "uri": "test_uri",
            "user": "test_user", 
            "password": "test_pass"
        }
        
        with patch('tools.milvus_base.MilvusClientWrapper') as mock_wrapper:
            mock_instance = MagicMock()
            mock_wrapper.return_value = mock_instance
            
            with tool._get_milvus_client(credentials) as client:
                assert client == mock_instance
    
    def test_validate_collection_name(self):
        """Test collection name validation"""
        tool = MilvusBaseTool()
        
        # Valid names
        assert tool._validate_collection_name("valid_collection") == True
        assert tool._validate_collection_name("collection123") == True
        
        # Invalid names  
        assert tool._validate_collection_name("123invalid") == False
        assert tool._validate_collection_name("invalid-name") == False
        assert tool._validate_collection_name("") == False
```

### Phase 3: Tool Implementation TDD

#### 3.1 Collection Tools Tests

**File:** `tests/unit/test_collection_tools.py`

```python
import pytest
from unittest.mock import patch, MagicMock
from tools.milvus_collection_create import MilvusCollectionCreateTool

class TestCollectionCreateTool:
    
    def test_create_simple_collection(self):
        """Test creating a simple vector collection"""
        tool = MilvusCollectionCreateTool()
        tool.runtime = MagicMock()
        
        params = {
            "collection_name": "test_collection",
            "dimension": 1536,
            "metric_type": "COSINE",
            "description": "Test collection"
        }
        
        with patch.object(tool, '_get_milvus_client') as mock_client_ctx:
            mock_client = MagicMock()
            mock_client_ctx.return_value.__enter__.return_value = mock_client
            mock_client.create_collection.return_value = {"status": "success"}
            
            # Execute tool
            results = list(tool._invoke(params))
            
            # Verify results
            assert len(results) == 1
            result = results[0].message
            assert result["success"] == True
            
            # Verify client was called correctly
            mock_client.create_collection.assert_called_once()
    
    def test_create_bm25_collection(self):
        """Test creating a collection with BM25 support"""
        tool = MilvusCollectionCreateTool()
        tool.runtime = MagicMock()
        
        params = {
            "collection_name": "bm25_collection",
            "dimension": 1536,
            "enable_bm25": True,
            "text_field_name": "content",
            "text_field_max_length": 5000
        }
        
        with patch.object(tool, '_get_milvus_client') as mock_client_ctx:
            mock_client = MagicMock()
            mock_client_ctx.return_value.__enter__.return_value = mock_client
            
            # Execute tool
            list(tool._invoke(params))
            
            # Verify BM25 schema was created
            call_args = mock_client.create_collection.call_args
            # Assert schema contains sparse vector field and BM25 function
```

#### 3.2 Search Tools Tests

**File:** `tests/unit/test_search_tools.py`

```python
from tools.milvus_search import MilvusSearchTool
from tools.milvus_bm25_search import MilvusBM25SearchTool

class TestVectorSearchTool:
    
    def test_vector_search_success(self):
        """Test successful vector search"""
        tool = MilvusSearchTool()
        tool.runtime = MagicMock()
        
        params = {
            "collection_name": "test_collection",
            "query_vector": "[0.1, 0.2, 0.3]",  # JSON string
            "limit": 5,
            "output_fields": "title,content"
        }
        
        with patch.object(tool, '_get_milvus_client') as mock_client_ctx:
            mock_client = MagicMock()
            mock_client_ctx.return_value.__enter__.return_value = mock_client
            
            # Mock search results
            mock_client.search.return_value = [
                [{"id": 1, "distance": 0.8, "entity": {"title": "Test"}}]
            ]
            
            results = list(tool._invoke(params))
            assert len(results) == 1
            
            # Verify search was called with correct parameters
            mock_client.search.assert_called_once()

class TestBM25SearchTool:
    
    def test_bm25_search_with_sparse_vector(self):
        """Test BM25 search on collection with sparse vector"""
        tool = MilvusBM25SearchTool()
        tool.runtime = MagicMock()
        
        params = {
            "collection_name": "bm25_collection",
            "query_text": "artificial intelligence",
            "limit": 3
        }
        
        with patch.object(tool, '_get_milvus_client') as mock_client_ctx:
            mock_client = MagicMock()
            mock_client_ctx.return_value.__enter__.return_value = mock_client
            
            # Mock collection with sparse vector support
            mock_client.describe_collection.return_value = {
                "fields": [
                    {"name": "sparse_vector", "type": "SparseFloatVector"}
                ]
            }
            
            mock_client.search.return_value = [
                [{"id": 1, "distance": 2.5, "entity": {"content": "AI content"}}]
            ]
            
            results = list(tool._invoke(params))
            
            # Verify BM25 search was performed
            search_call = mock_client.search.call_args
            assert "BM25" in str(search_call)
    
    def test_bm25_fallback_to_text_query(self):
        """Test BM25 falls back to text query when no sparse vector"""
        tool = MilvusBM25SearchTool()
        tool.runtime = MagicMock()
        
        # Mock collection without sparse vector
        with patch.object(tool, '_get_milvus_client') as mock_client_ctx:
            mock_client = MagicMock()
            mock_client_ctx.return_value.__enter__.return_value = mock_client
            
            mock_client.describe_collection.return_value = {
                "fields": [
                    {"name": "title", "type": "VarChar"},
                    {"name": "content", "type": "VarChar"}
                ]
            }
            
            mock_client.query.return_value = [
                {"id": 1, "title": "AI Article", "content": "About AI"}
            ]
            
            params = {
                "collection_name": "text_collection",
                "query_text": "artificial intelligence",
                "text_field": "content"
            }
            
            results = list(tool._invoke(params))
            
            # Verify text query was used as fallback
            mock_client.query.assert_called_once()
            query_call = mock_client.query.call_args
            assert "like" in str(query_call)
```

### Phase 4: BM25 Native Implementation TDD

#### 4.1 BM25 Collection Creation Tests

**File:** `tests/unit/test_bm25_collection.py`

```python
from pymilvus import DataType, Function, FunctionType

class TestBM25CollectionCreation:
    
    def test_bm25_schema_creation(self):
        """Test BM25 collection schema creation"""
        from tools.collection_helpers import create_bm25_schema
        
        schema = create_bm25_schema(
            collection_name="bm25_test",
            text_field="content",
            text_max_length=5000,
            vector_dim=1536
        )
        
        # Verify schema has required fields
        field_names = [f.name for f in schema.fields]
        assert "content" in field_names
        assert "sparse_vector" in field_names
        assert "embedding" in field_names
        
        # Verify sparse vector field type
        sparse_field = next(f for f in schema.fields if f.name == "sparse_vector")
        assert sparse_field.dtype == DataType.SPARSE_FLOAT_VECTOR
        
        # Verify BM25 function exists
        assert len(schema.functions) == 1
        bm25_func = schema.functions[0]
        assert bm25_func.function_type == FunctionType.BM25
        assert "content" in bm25_func.input_field_names
        assert bm25_func.output_field_names == "sparse_vector"

#### 4.2 BM25 Search Tests

```python
class TestBM25Search:
    
    def test_bm25_text_to_sparse_conversion(self):
        """Test text query conversion to sparse vector search"""
        # This tests the core BM25 functionality
        pass
    
    def test_bm25_parameter_tuning(self):
        """Test BM25 k1 and b parameter effects"""
        # Test different parameter combinations
        pass
    
    def test_bm25_vs_vector_search_results(self):
        """Test BM25 vs vector search result differences"""
        # Comparative testing
        pass
```

## Integration Testing Strategy

### Integration Test Setup

**File:** `tests/integration/conftest.py`

```python
import pytest
from pymilvus import MilvusClient
import os

@pytest.fixture(scope="session")
def milvus_client():
    """Real Milvus client for integration tests"""
    client = MilvusClient(
        uri=os.getenv("MILVUS_URI", "https://milvus-api.roomwits.com"),
        user=os.getenv("MILVUS_USER", "root"),
        password=os.getenv("MILVUS_PASSWORD"),
        db_name="test_db"
    )
    yield client
    # Cleanup test collections

@pytest.fixture
def test_collection(milvus_client):
    """Create and cleanup test collection"""
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    
    # Create collection
    schema = create_test_schema()
    milvus_client.create_collection(collection_name, schema=schema)
    
    yield collection_name
    
    # Cleanup
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
```

### Integration Test Cases

**File:** `tests/integration/test_collection_workflows.py`

```python
def test_complete_collection_lifecycle(milvus_client):
    """Test complete collection create→insert→search→delete workflow"""
    # Create collection with BM25 support
    # Insert test data
    # Perform searches
    # Verify results
    # Cleanup
    pass

def test_bm25_end_to_end(milvus_client):
    """Test BM25 functionality end-to-end"""
    # Create BM25-enabled collection
    # Insert text documents
    # Perform BM25 searches
    # Verify ranking and relevance
    pass
```

## Test Execution Strategy

### Local Development
```bash
# Run unit tests (fast feedback)
pytest tests/unit/ -v

# Run integration tests (slower, needs Milvus)
pytest tests/integration/ -v --milvus-uri=https://milvus-api.roomwits.com

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=tools --cov=provider --cov-report=html
```

### CI/CD Pipeline
```yaml
# .github/workflows/test.yml
name: Test Plugin
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.12
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r test_requirements.txt
      - name: Run unit tests
        run: pytest tests/unit/ -v
  
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v2
      - name: Setup Milvus
        run: |
          # Setup test Milvus instance
      - name: Run integration tests
        run: pytest tests/integration/ -v
        env:
          MILVUS_URI: ${{ secrets.TEST_MILVUS_URI }}
          MILVUS_USER: ${{ secrets.TEST_MILVUS_USER }}
          MILVUS_PASSWORD: ${{ secrets.TEST_MILVUS_PASSWORD }}
```

## Test Data Management

### Test Fixtures
```python
# tests/fixtures/test_data.py
TEST_DOCUMENTS = [
    {
        "id": 1,
        "title": "Artificial Intelligence Basics",
        "content": "AI is the simulation of human intelligence...",
        "embedding": [0.1, 0.2, 0.3, ...]  # 1536 dimensions
    },
    # More test documents
]

TEST_BM25_QUERIES = [
    {"query": "artificial intelligence", "expected_top_result": 1},
    {"query": "machine learning", "expected_top_result": 2},
    # More test queries
]
```

## Performance Testing

### Benchmark Tests
```python
# tests/performance/test_benchmarks.py
def test_search_performance():
    """Benchmark search performance vs HTTP implementation"""
    # Time vector searches
    # Time BM25 searches  
    # Compare with baseline
    pass

def test_connection_performance():
    """Test connection establishment time"""
    # Test PyMilvus vs HTTP connection times
    pass
```

## Quality Metrics

### Code Coverage Targets
- **Unit Tests:** 90%+ coverage
- **Integration Tests:** 80%+ coverage
- **Critical Path:** 100% coverage (auth, search, BM25)

### Test Quality Metrics
- **Test Speed:** Unit tests < 50ms each
- **Test Isolation:** No test dependencies
- **Test Reliability:** < 1% flaky test rate

## TDD Benefits for This Refactoring

1. **Confidence:** Tests prove PyMilvus implementation works
2. **Regression Prevention:** Ensure no features break during migration
3. **Documentation:** Tests document expected behavior
4. **Design Quality:** TDD leads to better API design
5. **Refactoring Safety:** Can refactor with confidence

## Next Steps

1. **Setup Test Infrastructure**
   - Create test directory structure
   - Setup pytest configuration
   - Create base test fixtures

2. **Start Phase 1 TDD**
   - Write provider validation tests
   - Implement provider changes
   - Verify tests pass

3. **Continue Phase by Phase**
   - Follow Red-Green-Refactor for each component
   - Maintain test coverage above targets
   - Run full test suite before each phase completion