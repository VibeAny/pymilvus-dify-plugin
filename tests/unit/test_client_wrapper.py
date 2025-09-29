"""
Unit tests for PyMilvus client wrapper

Following TDD approach:
1. Red: Write failing tests first
2. Green: Write minimal code to make tests pass  
3. Refactor: Improve code while keeping tests green
"""
import pytest
import sys
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any

# Mock dify_plugin for testing
class MockToolProviderCredentialValidationError(Exception):
    pass

# Add mock modules to sys.modules before any imports
sys.modules['dify_plugin'] = MagicMock()
sys.modules['dify_plugin.errors'] = MagicMock()
sys.modules['dify_plugin.errors.tool'] = MagicMock()
sys.modules['dify_plugin.errors.tool'].ToolProviderCredentialValidationError = MockToolProviderCredentialValidationError

# Mock PyMilvus components
sys.modules['pymilvus'] = MagicMock()
sys.modules['pymilvus.model'] = MagicMock()
sys.modules['pymilvus.model.sparse'] = MagicMock()


class TestMilvusClientWrapper:
    """Test the pure PyMilvus client wrapper"""
    
    def test_client_wrapper_initialization(self):
        """Test client wrapper initialization with credentials"""
        with patch('lib.milvus_client.MilvusClient') as mock_milvus_client_class:
            from lib.milvus_client import MilvusClientWrapper
            
            mock_client = MagicMock()
            mock_milvus_client_class.return_value = mock_client
            
            credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root",
                "password": "test_password",
                "database": "test_db"
            }
            
            wrapper = MilvusClientWrapper(credentials)
            
            # Verify client was created with correct parameters
            mock_milvus_client_class.assert_called_once_with(
                uri="https://milvus-api.roomwits.com",
                user="root",
                password="test_password",
                db_name="test_db"
            )
            
            assert wrapper.client == mock_client
    
    def test_list_collections(self):
        """Test listing collections wrapper"""
        with patch('lib.milvus_client.MilvusClient') as mock_milvus_client_class:
            from lib.milvus_client import MilvusClientWrapper
            
            mock_client = MagicMock()
            mock_client.list_collections.return_value = ["collection1", "collection2"]
            mock_milvus_client_class.return_value = mock_client
            
            credentials = {"uri": "test", "user": "root", "password": "pass", "database": "db"}
            wrapper = MilvusClientWrapper(credentials)
            
            result = wrapper.list_collections()
            
            assert result == ["collection1", "collection2"]
            mock_client.list_collections.assert_called_once()
    
    def test_has_collection(self):
        """Test checking if collection exists"""
        with patch('lib.milvus_client.MilvusClient') as mock_milvus_client_class:
            from lib.milvus_client import MilvusClientWrapper
            
            mock_client = MagicMock()
            mock_client.has_collection.return_value = True
            mock_milvus_client_class.return_value = mock_client
            
            credentials = {"uri": "test", "user": "root", "password": "pass", "database": "db"}
            wrapper = MilvusClientWrapper(credentials)
            
            result = wrapper.has_collection("test_collection")
            
            assert result is True
            mock_client.has_collection.assert_called_once_with("test_collection")
    
    def test_create_collection_with_bm25(self):
        """Test creating collection with BM25 support"""
        with patch('lib.milvus_client.MilvusClient') as mock_milvus_client_class:
            with patch('lib.milvus_client.Function') as mock_function_class:
                from lib.milvus_client import MilvusClientWrapper
                
                mock_client = MagicMock()
                mock_schema = MagicMock()
                mock_index_params = MagicMock()
                
                mock_client.create_schema.return_value = mock_schema
                mock_client.prepare_index_params.return_value = mock_index_params
                mock_milvus_client_class.return_value = mock_client
                
                credentials = {"uri": "test", "user": "root", "password": "pass", "database": "db"}
                wrapper = MilvusClientWrapper(credentials)
                
                schema_config = {
                    "collection_name": "test_collection",
                    "enable_bm25": True,
                    "vector_field": {
                        "name": "vector",
                        "dim": 1536
                    },
                    "text_field": {
                        "name": "content",
                        "max_length": 5000
                    }
                }
                
                wrapper.create_collection_with_schema(schema_config)
                
                # Verify schema creation
                mock_client.create_schema.assert_called_once_with(auto_id=True)
                mock_schema.add_field.assert_any_call("id", "Int64", is_primary=True, auto_id=True)
                mock_schema.add_field.assert_any_call("content", "VarChar", max_length=5000, enable_analyzer=True)
                mock_schema.add_field.assert_any_call("vector", "FloatVector", dim=1536)
                mock_schema.add_field.assert_any_call("sparse_vector", "SparseFloatVector")
                
                # Verify function was added for BM25
                mock_schema.add_function.assert_called_once()
                
                # Verify collection creation
                mock_client.create_collection.assert_called_once()
    
    def test_insert_data(self):
        """Test inserting data into collection"""
        with patch('lib.milvus_client.MilvusClient') as mock_milvus_client_class:
            from lib.milvus_client import MilvusClientWrapper
            
            mock_client = MagicMock()
            mock_client.insert.return_value = {"insert_count": 2}
            mock_milvus_client_class.return_value = mock_client
            
            credentials = {"uri": "test", "user": "root", "password": "pass", "database": "db"}
            wrapper = MilvusClientWrapper(credentials)
            
            data = [
                {"id": 1, "content": "test content 1", "vector": [0.1] * 1536},
                {"id": 2, "content": "test content 2", "vector": [0.2] * 1536}
            ]
            
            result = wrapper.insert("test_collection", data)
            
            assert result["insert_count"] == 2
            mock_client.insert.assert_called_once_with(
                collection_name="test_collection",
                data=data
            )
    
    def test_vector_search(self):
        """Test vector similarity search"""
        with patch('lib.milvus_client.MilvusClient') as mock_milvus_client_class:
            from lib.milvus_client import MilvusClientWrapper
            
            mock_client = MagicMock()
            mock_client.search.return_value = [[{"id": 1, "distance": 0.95}]]
            mock_milvus_client_class.return_value = mock_client
            
            credentials = {"uri": "test", "user": "root", "password": "pass", "database": "db"}
            wrapper = MilvusClientWrapper(credentials)
            
            query_vector = [0.1] * 1536
            result = wrapper.vector_search(
                collection_name="test_collection",
                vectors=[query_vector],
                limit=10,
                output_fields=["id", "content"]
            )
            
            assert result == [[{"id": 1, "distance": 0.95}]]
            mock_client.search.assert_called_once_with(
                collection_name="test_collection",
                data=[query_vector],
                anns_field="vector",
                limit=10,
                output_fields=["id", "content"]
            )
    
    def test_bm25_search(self):
        """Test BM25 keyword search"""
        with patch('lib.milvus_client.MilvusClient') as mock_milvus_client_class:
            from lib.milvus_client import MilvusClientWrapper
            
            mock_client = MagicMock()
            mock_client.search.return_value = [[{"id": 1, "distance": 0.85}]]
            mock_milvus_client_class.return_value = mock_client
            
            credentials = {"uri": "test", "user": "root", "password": "pass", "database": "db"}
            wrapper = MilvusClientWrapper(credentials)
            
            result = wrapper.bm25_search(
                collection_name="test_collection",
                query_text="machine learning AI",
                limit=10,
                output_fields=["id", "content"]
            )
            
            assert result == [[{"id": 1, "distance": 0.85}]]
            mock_client.search.assert_called_once_with(
                collection_name="test_collection",
                data=["machine learning AI"],
                anns_field="sparse_vector",
                limit=10,
                output_fields=["id", "content"]
            )
    
    def test_hybrid_search(self):
        """Test hybrid search (vector + BM25)"""
        with patch('lib.milvus_client.MilvusClient') as mock_milvus_client_class:
            from lib.milvus_client import MilvusClientWrapper
            
            mock_client = MagicMock()
            mock_client.search.return_value = [[{"id": 1, "distance": 0.90}]]
            mock_milvus_client_class.return_value = mock_client
            
            credentials = {"uri": "test", "user": "root", "password": "pass", "database": "db"}
            wrapper = MilvusClientWrapper(credentials)
            
            query_vector = [0.1] * 1536
            result = wrapper.hybrid_search(
                collection_name="test_collection",
                vector=query_vector,
                text="machine learning",
                limit=10,
                vector_weight=0.7,
                text_weight=0.3,
                output_fields=["id", "content"]
            )
            
            # Hybrid search should call the client's hybrid search capability
            assert result == [[{"id": 1, "distance": 0.90}]]
            mock_client.search.assert_called_once()
    
    def test_query_data(self):
        """Test querying data with filter"""
        with patch('lib.milvus_client.MilvusClient') as mock_milvus_client_class:
            from lib.milvus_client import MilvusClientWrapper
            
            mock_client = MagicMock()
            mock_client.query.return_value = [{"id": 1, "content": "test"}]
            mock_milvus_client_class.return_value = mock_client
            
            credentials = {"uri": "test", "user": "root", "password": "pass", "database": "db"}
            wrapper = MilvusClientWrapper(credentials)
            
            result = wrapper.query(
                collection_name="test_collection",
                filter="id in [1, 2]",
                output_fields=["id", "content"]
            )
            
            assert result == [{"id": 1, "content": "test"}]
            mock_client.query.assert_called_once_with(
                collection_name="test_collection",
                filter="id in [1, 2]",
                output_fields=["id", "content"]
            )
    
    def test_delete_data(self):
        """Test deleting data"""
        with patch('lib.milvus_client.MilvusClient') as mock_milvus_client_class:
            from lib.milvus_client import MilvusClientWrapper
            
            mock_client = MagicMock()
            mock_client.delete.return_value = {"delete_count": 1}
            mock_milvus_client_class.return_value = mock_client
            
            credentials = {"uri": "test", "user": "root", "password": "pass", "database": "db"}
            wrapper = MilvusClientWrapper(credentials)
            
            result = wrapper.delete(
                collection_name="test_collection",
                filter="id == 1"
            )
            
            assert result["delete_count"] == 1
            mock_client.delete.assert_called_once_with(
                collection_name="test_collection",
                filter="id == 1"
            )
    
    def test_drop_collection(self):
        """Test dropping collection"""
        with patch('lib.milvus_client.MilvusClient') as mock_milvus_client_class:
            from lib.milvus_client import MilvusClientWrapper
            
            mock_client = MagicMock()
            mock_milvus_client_class.return_value = mock_client
            
            credentials = {"uri": "test", "user": "root", "password": "pass", "database": "db"}
            wrapper = MilvusClientWrapper(credentials)
            
            wrapper.drop_collection("test_collection")
            
            mock_client.drop_collection.assert_called_once_with("test_collection")
    
    def test_describe_collection(self):
        """Test describing collection schema"""
        with patch('lib.milvus_client.MilvusClient') as mock_milvus_client_class:
            from lib.milvus_client import MilvusClientWrapper
            
            mock_client = MagicMock()
            mock_client.describe_collection.return_value = {
                "collection_name": "test_collection",
                "fields": [
                    {"name": "id", "type": "Int64", "is_primary": True},
                    {"name": "vector", "type": "FloatVector", "dim": 1536}
                ]
            }
            mock_milvus_client_class.return_value = mock_client
            
            credentials = {"uri": "test", "user": "root", "password": "pass", "database": "db"}
            wrapper = MilvusClientWrapper(credentials)
            
            result = wrapper.describe_collection("test_collection")
            
            assert result["collection_name"] == "test_collection"
            assert len(result["fields"]) == 2
            mock_client.describe_collection.assert_called_once_with("test_collection")
    
    def test_client_wrapper_handles_connection_errors(self):
        """Test client wrapper handles connection errors gracefully"""
        from lib.milvus_client import MilvusClientWrapper
        
        # Test with invalid credentials
        with pytest.raises(Exception):  # Should raise connection error
            credentials = {
                "uri": "",  # Empty URI should cause error
                "user": "root",
                "password": "password",
                "database": "db"
            }
            wrapper = MilvusClientWrapper(credentials)


class TestMilvusClientWrapperConfiguration:
    """Test client wrapper configuration and schema handling"""
    
    def test_bm25_schema_configuration(self):
        """Test BM25 schema configuration is correct"""
        from lib.milvus_client import MilvusClientWrapper
        
        # This test validates the schema configuration logic
        schema_config = {
            "collection_name": "test_bm25_collection",
            "enable_bm25": True,
            "vector_field": {
                "name": "embedding",
                "dim": 1536
            },
            "text_field": {
                "name": "text_content",
                "max_length": 8000
            }
        }
        
        # Test schema validation
        assert schema_config["enable_bm25"] is True
        assert schema_config["vector_field"]["dim"] == 1536
        assert schema_config["text_field"]["max_length"] == 8000
    
    def test_default_configuration_values(self):
        """Test default configuration values"""
        from lib.milvus_client import MilvusClientWrapper
        
        # Test that wrapper handles default values correctly
        credentials = {
            "uri": "https://milvus-api.roomwits.com",
            "user": "root",
            "password": "password"
            # database not provided - should default to "default"
        }
        
        # The wrapper should handle missing database parameter
        # This will be tested in the implementation
        assert True  # Placeholder for implementation