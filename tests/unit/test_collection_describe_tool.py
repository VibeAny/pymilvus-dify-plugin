"""
Unit tests for milvus_collection_describe tool

Following TDD approach:
1. Red: Write failing tests first
2. Green: Write minimal code to make tests pass  
3. Refactor: Improve code while keeping tests green
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any


class TestMilvusCollectionDescribeTool:
    """Test the milvus_collection_describe tool using TDD"""
    
    def test_collection_describe_tool_initialization(self):
        """Test that the tool can be initialized properly"""
        with patch('tools.milvus_collection_describe.MilvusBaseTool'):
            from tools.milvus_collection_describe import MilvusCollectionDescribeTool
            
            tool = MilvusCollectionDescribeTool()
            assert tool is not None
            assert hasattr(tool, '_invoke')
            assert hasattr(tool, 'base_tool')  # Composition pattern
    
    def test_collection_describe_success(self):
        """Test successful collection description"""
        with patch('tools.milvus_collection_describe.MilvusBaseTool') as mock_base_tool_class:
            # Mock the base tool instance
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            # Mock the client context manager
            mock_client = MagicMock()
            mock_description = {
                "collection_name": "test_collection",
                "dimension": 1536,
                "metric_type": "COSINE",
                "fields": [
                    {"name": "id", "type": "INT64", "is_primary": True},
                    {"name": "vector", "type": "FLOAT_VECTOR", "dimension": 1536}
                ]
            }
            mock_client.has_collection.return_value = True
            mock_client.describe_collection.return_value = mock_description
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_collection_describe import MilvusCollectionDescribeTool
            
            tool = MilvusCollectionDescribeTool()
            tool.runtime = MagicMock()
            tool.runtime.credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root", 
                "password": "test_password",
                "database": "test_db"
            }
            
            # Mock create_json_message
            def create_json_message(data):
                mock_msg = MagicMock()
                mock_msg.message = data
                return mock_msg
            tool.create_json_message = create_json_message
            
            tool_parameters = {"collection_name": "test_collection"}
            
            # Execute the tool
            result = list(tool._invoke(tool_parameters))
            
            # Verify results
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is True
            assert response["collection_name"] == "test_collection"
            assert "description" in response
            assert response["description"] == mock_description
            
            # Verify client was called correctly
            mock_client.has_collection.assert_called_once_with("test_collection")
            mock_client.describe_collection.assert_called_once_with("test_collection")
    
    def test_collection_describe_missing_name(self):
        """Test error when collection name is missing"""
        with patch('tools.milvus_collection_describe.MilvusBaseTool') as mock_base_tool_class:
            # Mock the base tool instance
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            from tools.milvus_collection_describe import MilvusCollectionDescribeTool
            
            tool = MilvusCollectionDescribeTool()
            tool.runtime = MagicMock()
            tool.runtime.credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root",
                "password": "test_password",
                "database": "test_db"
            }
            
            # Mock create_json_message
            def create_json_message(data):
                mock_msg = MagicMock()
                mock_msg.message = data
                return mock_msg
            tool.create_json_message = create_json_message
            
            tool_parameters = {}  # Missing collection_name
            
            # Execute the tool
            result = list(tool._invoke(tool_parameters))
            
            # Verify error response
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "error" in response
            assert "collection_name is required" in response["error"].lower()
    
    def test_collection_describe_not_exists(self):
        """Test error when collection does not exist"""
        with patch('tools.milvus_collection_describe.MilvusBaseTool') as mock_base_tool_class:
            # Mock the base tool instance
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            # Mock the client context manager
            mock_client = MagicMock()
            mock_client.has_collection.return_value = False
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_collection_describe import MilvusCollectionDescribeTool
            
            tool = MilvusCollectionDescribeTool()
            tool.runtime = MagicMock()
            tool.runtime.credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root",
                "password": "test_password",
                "database": "test_db"
            }
            
            # Mock create_json_message
            def create_json_message(data):
                mock_msg = MagicMock()
                mock_msg.message = data
                return mock_msg
            tool.create_json_message = create_json_message
            
            tool_parameters = {"collection_name": "nonexistent_collection"}
            
            # Execute the tool
            result = list(tool._invoke(tool_parameters))
            
            # Verify error response
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "error" in response
            assert "does not exist" in response["error"]
    
    def test_collection_describe_connection_error(self):
        """Test handling of connection errors"""
        with patch('tools.milvus_collection_describe.MilvusBaseTool') as mock_base_tool_class:
            # Mock the base tool instance
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            # Mock the client context manager to raise exception
            mock_base_tool_instance._get_milvus_client.side_effect = Exception("Connection failed")
            
            from tools.milvus_collection_describe import MilvusCollectionDescribeTool
            
            tool = MilvusCollectionDescribeTool()
            tool.runtime = MagicMock()
            tool.runtime.credentials = {
                "uri": "invalid_uri",
                "user": "root",
                "password": "test_password",
                "database": "test_db"
            }
            
            # Mock create_json_message
            def create_json_message(data):
                mock_msg = MagicMock()
                mock_msg.message = data
                return mock_msg
            tool.create_json_message = create_json_message
            
            tool_parameters = {"collection_name": "test_collection"}
            
            # Execute the tool
            result = list(tool._invoke(tool_parameters))
            
            # Verify error response
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "error" in response
            assert "Connection failed" in response["error"]


class TestMilvusCollectionDescribeToolConfiguration:
    """Test the tool configuration aspects"""
    
    def test_tool_yaml_structure_expectations(self):
        """Test expectations for the tool YAML configuration"""
        # This test validates what we expect the YAML to contain
        expected_yaml_structure = {
            "identity": {
                "name": "milvus_collection_describe",
                "author": "vibeany",
                "label": {"en_US": "Describe Milvus Collection"}
            },
            "description": {
                "en_US": "Get detailed information about a Milvus collection schema and configuration"
            },
            "parameters": [
                {
                    "name": "collection_name",
                    "type": "string",
                    "required": True,
                    "label": {"en_US": "Collection Name"}
                }
            ],
            "extra": {
                "python": {
                    "source": "tools/milvus_collection_describe.py"
                }
            }
        }
        
        # Verify the structure is sensible
        assert expected_yaml_structure["identity"]["name"] == "milvus_collection_describe"
        assert len(expected_yaml_structure["parameters"]) == 1
        assert expected_yaml_structure["parameters"][0]["name"] == "collection_name"
        assert expected_yaml_structure["parameters"][0]["required"] is True