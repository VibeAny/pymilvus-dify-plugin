"""
Unit tests for refactored Milvus base tool class using PyMilvus

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


class TestMilvusBaseTool:
    """Test the refactored PyMilvus-based base tool class"""
    
    def test_get_milvus_client_context_manager(self):
        """Test context manager creates PyMilvus client properly"""
        with patch('lib.milvus_client.MilvusClient'):
            from tools.milvus_base import MilvusBaseTool
            
            base_tool = MilvusBaseTool()
            credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root",
                "password": "test_password",
                "database": "test_db"
            }
            
            with base_tool._get_milvus_client(credentials) as client:
                assert client is not None
                # Should return our MilvusClientWrapper instance
                assert hasattr(client, 'list_collections')
                assert hasattr(client, 'has_collection')
                assert hasattr(client, 'insert')
                assert hasattr(client, 'search')
    
    def test_get_milvus_client_handles_missing_credentials(self):
        """Test context manager handles missing credentials properly"""
        from tools.milvus_base import MilvusBaseTool
        
        base_tool = MilvusBaseTool()
        
        # Test missing URI
        with pytest.raises(ValueError, match="URI is required"):
            with base_tool._get_milvus_client({"user": "root", "password": "pass"}):
                pass
        
        # Test missing user
        with pytest.raises(ValueError, match="Username is required"):
            with base_tool._get_milvus_client({"uri": "test", "password": "pass"}):
                pass
        
        # Test missing password
        with pytest.raises(ValueError, match="Password is required"):
            with base_tool._get_milvus_client({"uri": "test", "user": "root"}):
                pass
    
    def test_get_milvus_client_handles_connection_errors(self):
        """Test context manager handles connection errors properly"""
        with patch('tools.milvus_base.MilvusClientWrapper') as mock_wrapper:
            mock_wrapper.side_effect = Exception("Connection failed")
            
            from tools.milvus_base import MilvusBaseTool
            
            base_tool = MilvusBaseTool()
            credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root",
                "password": "test_password",
                "database": "test_db"
            }
            
            with pytest.raises(ValueError, match="Failed to connect to Milvus"):
                with base_tool._get_milvus_client(credentials):
                    pass
    
    def test_validate_collection_name(self):
        """Test collection name validation"""
        from tools.milvus_base import MilvusBaseTool
        
        base_tool = MilvusBaseTool()
        
        # Valid names
        assert base_tool._validate_collection_name("valid_collection")
        assert base_tool._validate_collection_name("collection123")
        assert base_tool._validate_collection_name("_collection")
        assert base_tool._validate_collection_name("Collection_Name")
        
        # Invalid names
        assert not base_tool._validate_collection_name("")
        assert not base_tool._validate_collection_name("123collection")  # starts with number
        assert not base_tool._validate_collection_name("collection-name")  # has hyphen
        assert not base_tool._validate_collection_name("collection.name")  # has dot
        assert not base_tool._validate_collection_name("collection name")  # has space
        assert not base_tool._validate_collection_name("a" * 256)  # too long
        assert not base_tool._validate_collection_name(None)  # not string
        assert not base_tool._validate_collection_name(123)  # not string
    
    def test_parse_vector_data(self):
        """Test vector data parsing"""
        from tools.milvus_base import MilvusBaseTool
        
        base_tool = MilvusBaseTool()
        
        # Test valid JSON string
        result = base_tool._parse_vector_data('[0.1, 0.2, 0.3]')
        assert result == [0.1, 0.2, 0.3]
        
        # Test already parsed list
        result = base_tool._parse_vector_data([0.1, 0.2, 0.3])
        assert result == [0.1, 0.2, 0.3]
        
        # Test invalid JSON
        with pytest.raises(ValueError, match="Invalid vector data format"):
            base_tool._parse_vector_data("invalid json")
        
        # Test non-string, non-list
        with pytest.raises(ValueError, match="Invalid vector data format"):
            base_tool._parse_vector_data(123)
    
    def test_parse_search_params(self):
        """Test search parameters parsing"""
        from tools.milvus_base import MilvusBaseTool
        
        base_tool = MilvusBaseTool()
        
        # Test valid JSON string
        result = base_tool._parse_search_params('{"nprobe": 10, "ef": 64}')
        assert result == {"nprobe": 10, "ef": 64}
        
        # Test empty/None input
        result = base_tool._parse_search_params(None)
        assert result == {}
        
        result = base_tool._parse_search_params("")
        assert result == {}
        
        # Test invalid JSON (should return empty dict, not raise)
        result = base_tool._parse_search_params("invalid json")
        assert result == {}
    
    def test_format_error_message(self):
        """Test error message formatting"""
        from tools.milvus_base import MilvusBaseTool
        
        base_tool = MilvusBaseTool()
        
        # Test standard exception
        error = Exception("Test error message")
        result = base_tool._format_error_message(error)
        assert "Test error message" in result
        
        # Test with operation context
        result = base_tool._format_error_message(error, operation="collection creation")
        assert "collection creation" in result
        assert "Test error message" in result
    
    def test_logging_configuration(self):
        """Test that logging is properly configured"""
        from tools.milvus_base import MilvusBaseTool
        
        # Verify logger is set up
        import logging
        logger = logging.getLogger('tools.milvus_base')
        assert logger is not None


class TestMilvusBaseToolIntegration:
    """Integration tests for MilvusBaseTool with client wrapper"""
    
    def test_client_wrapper_integration(self):
        """Test that base tool integrates properly with client wrapper"""
        with patch('lib.milvus_client.MilvusClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.list_collections.return_value = ["test_collection"]
            mock_client_class.return_value = mock_client
            
            from tools.milvus_base import MilvusBaseTool
            
            base_tool = MilvusBaseTool()
            credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root",
                "password": "test_password",
                "database": "test_db"
            }
            
            with base_tool._get_milvus_client(credentials) as client:
                # Test that we can call client methods
                collections = client.list_collections()
                assert collections == ["test_collection"]
                
                # Verify the client wrapper was initialized correctly
                mock_client_class.assert_called_once()
    
    def test_credentials_passed_correctly(self):
        """Test that credentials are passed correctly to client wrapper"""
        with patch('tools.milvus_base.MilvusClientWrapper') as mock_wrapper:
            mock_client = MagicMock()
            mock_wrapper.return_value = mock_client
            
            from tools.milvus_base import MilvusBaseTool
            
            base_tool = MilvusBaseTool()
            credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "test_user",
                "password": "test_password",
                "database": "custom_db"
            }
            
            with base_tool._get_milvus_client(credentials) as client:
                # Verify wrapper was called with correct credentials
                mock_wrapper.assert_called_once_with(credentials)
    
    def test_default_database_handling(self):
        """Test default database parameter handling"""
        with patch('tools.milvus_base.MilvusClientWrapper') as mock_wrapper:
            mock_client = MagicMock()
            mock_wrapper.return_value = mock_client
            
            from tools.milvus_base import MilvusBaseTool
            
            base_tool = MilvusBaseTool()
            credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "test_user",
                "password": "test_password"
                # No database specified
            }
            
            with base_tool._get_milvus_client(credentials) as client:
                # Should add default database
                expected_credentials = credentials.copy()
                expected_credentials["database"] = "default"
                mock_wrapper.assert_called_once_with(expected_credentials)