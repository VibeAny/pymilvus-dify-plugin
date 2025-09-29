"""
Test configuration and fixtures for Milvus Plugin rewrite tests
"""
import pytest
import os
import sys
from unittest.mock import MagicMock, patch
from typing import Dict, Any

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class MockToolProviderCredentialValidationError(Exception):
    pass

class MockToolProvider:
    pass

class MockTool:
    pass

class MockToolInvokeMessage:
    pass

sys.modules['dify_plugin'] = MagicMock()
sys.modules['dify_plugin.errors'] = MagicMock()
sys.modules['dify_plugin.errors.tool'] = MagicMock()
sys.modules['dify_plugin.entities'] = MagicMock()
sys.modules['dify_plugin.entities.tool'] = MagicMock()

sys.modules['dify_plugin'].ToolProvider = MockToolProvider
sys.modules['dify_plugin'].Tool = MockTool
sys.modules['dify_plugin.errors.tool'].ToolProviderCredentialValidationError = MockToolProviderCredentialValidationError
sys.modules['dify_plugin.entities.tool'].ToolInvokeMessage = MockToolInvokeMessage

# Test configuration
TEST_MILVUS_URI = os.getenv("TEST_MILVUS_URI", "https://milvus-api.roomwits.com")
TEST_MILVUS_USER = os.getenv("TEST_MILVUS_USER", "root")
TEST_MILVUS_PASSWORD = os.getenv("TEST_MILVUS_PASSWORD", "test_password")
TEST_MILVUS_DATABASE = os.getenv("TEST_MILVUS_DATABASE", "test_db")

@pytest.fixture
def mock_credentials():
    """Standard test credentials - simplified for Dify model integration"""
    return {
        "uri": TEST_MILVUS_URI,
        "user": TEST_MILVUS_USER,
        "password": TEST_MILVUS_PASSWORD,
        "database": TEST_MILVUS_DATABASE
    }

@pytest.fixture
def mock_milvus_client():
    """Mock PyMilvus client for unit tests"""
    with patch('pymilvus.MilvusClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Configure default mock behaviors
        mock_client.list_collections.return_value = ["test_collection"]
        mock_client.has_collection.return_value = True
        mock_client.describe_collection.return_value = {
            "collection_name": "test_collection",
            "fields": [
                {"name": "id", "type": "Int64", "is_primary": True},
                {"name": "vector", "type": "FloatVector", "dim": 1536}
            ]
        }
        mock_client.insert.return_value = {"insert_count": 1}
        mock_client.search.return_value = [[{"id": 1, "distance": 0.8}]]
        mock_client.query.return_value = [{"id": 1, "content": "test"}]
        
        yield mock_client

@pytest.fixture
def mock_tool_runtime():
    """Mock tool runtime for testing tools"""
    runtime = MagicMock()
    runtime.credentials = {
        "uri": TEST_MILVUS_URI,
        "user": TEST_MILVUS_USER,
        "password": TEST_MILVUS_PASSWORD,
        "database": TEST_MILVUS_DATABASE
    }
    return runtime

@pytest.fixture
def sample_collection_schema():
    """Sample collection schema for testing"""
    return {
        "collection_name": "test_collection",
        "auto_id": True,
        "fields": [
            {
                "name": "id",
                "type": "Int64",
                "is_primary": True,
                "auto_id": True
            },
            {
                "name": "content",
                "type": "VarChar",
                "max_length": 5000,
                "enable_analyzer": True
            },
            {
                "name": "vector",
                "type": "FloatVector",
                "dim": 1536
            },
            {
                "name": "sparse_vector",
                "type": "SparseFloatVector"
            }
        ],
        "functions": [
            {
                "name": "bm25_fn",
                "type": "BM25",
                "input_fields": ["content"],
                "output_fields": "sparse_vector"
            }
        ]
    }

@pytest.fixture
def sample_test_data():
    """Sample test data for insertion tests"""
    return [
        {
            "id": 1,
            "content": "Artificial intelligence and machine learning",
            "vector": [0.1] * 1536,
            "metadata": {"category": "AI"}
        },
        {
            "id": 2,
            "content": "Natural language processing and text analysis", 
            "vector": [0.2] * 1536,
            "metadata": {"category": "NLP"}
        }
    ]

# Test markers configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, no external deps)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require real Milvus)"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests (full workflow)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location"""
    for item in items:
        # Auto-mark based on file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

@pytest.fixture(scope="session")
def real_milvus_client():
    """Real Milvus client for integration tests - only if credentials are available"""
    if not all([TEST_MILVUS_URI, TEST_MILVUS_USER, TEST_MILVUS_PASSWORD]):
        pytest.skip("Real Milvus credentials not available for integration tests")
    
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(
            uri=TEST_MILVUS_URI,
            user=TEST_MILVUS_USER,
            password=TEST_MILVUS_PASSWORD,
            db_name=TEST_MILVUS_DATABASE
        )
        
        # Test connection
        client.list_collections()
        yield client
        
    except Exception as e:
        pytest.skip(f"Cannot connect to real Milvus instance: {e}")

@pytest.fixture
def integration_test_collection(real_milvus_client):
    """Create and cleanup test collection for integration tests"""
    import uuid
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    
    try:
        # Create test collection
        schema = real_milvus_client.create_schema(auto_id=True)
        schema.add_field("id", "Int64", is_primary=True, auto_id=True)
        schema.add_field("vector", "FloatVector", dim=128)
        
        index_params = real_milvus_client.prepare_index_params()
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
        
        real_milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        
        yield collection_name
        
    finally:
        # Cleanup
        try:
            if real_milvus_client.has_collection(collection_name):
                real_milvus_client.drop_collection(collection_name)
        except Exception:
            pass  # Ignore cleanup errors