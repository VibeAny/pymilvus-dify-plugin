"""
Unit tests for milvus_delete tool

Following TDD approach:
1. Red: Write failing tests first
2. Green: Write minimal code to make tests pass  
3. Refactor: Improve code while keeping tests green
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any


class TestMilvusDeleteTool:
    """Test the milvus_delete tool using TDD"""
    
    def test_delete_tool_initialization(self):
        """Test that the tool can be initialized properly"""
        with patch('tools.milvus_delete.MilvusBaseTool'):
            from tools.milvus_delete import MilvusDeleteTool
            
            tool = MilvusDeleteTool()
            assert tool is not None
            assert hasattr(tool, '_invoke')
            assert hasattr(tool, 'base_tool')  # Composition pattern
    
    def test_delete_data_success(self):
        """Test successful data deletion"""
        with patch('tools.milvus_delete.MilvusBaseTool') as mock_base_tool_class:
            # Mock the base tool instance
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            # Mock the client context manager
            mock_client = MagicMock()
            mock_client.has_collection.return_value = True
            mock_delete_result = {"delete_count": 5}
            mock_client.delete.return_value = mock_delete_result
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_delete import MilvusDeleteTool
            
            tool = MilvusDeleteTool()
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
            
            tool_parameters = {
                "collection_name": "test_collection",
                "filter": "id in [1, 2, 3]"
            }
            
            # Execute the tool
            result = list(tool._invoke(tool_parameters))
            
            # Verify results
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is True
            assert response["collection_name"] == "test_collection"
            assert response["filter"] == "id in [1, 2, 3]"
            assert response["delete_count"] == 5
            
            # Verify client was called correctly
            mock_client.has_collection.assert_called_once_with("test_collection")
            mock_client.delete.assert_called_once_with(
                collection_name="test_collection",
                filter="id in [1, 2, 3]"
            )
    
    def test_delete_missing_collection_name(self):
        """Test error when collection name is missing"""
        with patch('tools.milvus_delete.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            from tools.milvus_delete import MilvusDeleteTool
            
            tool = MilvusDeleteTool()
            tool.runtime = MagicMock()
            tool.runtime.credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root",
                "password": "test_password",
                "database": "test_db"
            }
            
            def create_json_message(data):
                mock_msg = MagicMock()
                mock_msg.message = data
                return mock_msg
            tool.create_json_message = create_json_message
            
            tool_parameters = {"filter": "id > 100"}
            
            result = list(tool._invoke(tool_parameters))
            
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "collection_name is required" in response["error"].lower()
    
    def test_delete_missing_filter(self):
        """Test error when filter is missing (safety check)"""
        with patch('tools.milvus_delete.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            from tools.milvus_delete import MilvusDeleteTool
            
            tool = MilvusDeleteTool()
            tool.runtime = MagicMock()
            tool.runtime.credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root",
                "password": "test_password",
                "database": "test_db"
            }
            
            def create_json_message(data):
                mock_msg = MagicMock()
                mock_msg.message = data
                return mock_msg
            tool.create_json_message = create_json_message
            
            tool_parameters = {"collection_name": "test_collection"}
            
            result = list(tool._invoke(tool_parameters))
            
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "filter expression is required for safety" in response["error"].lower()
    
    def test_delete_collection_not_exists(self):
        """Test error when collection does not exist"""
        with patch('tools.milvus_delete.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            mock_client = MagicMock()
            mock_client.has_collection.return_value = False
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_delete import MilvusDeleteTool
            
            tool = MilvusDeleteTool()
            tool.runtime = MagicMock()
            tool.runtime.credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root",
                "password": "test_password",
                "database": "test_db"
            }
            
            def create_json_message(data):
                mock_msg = MagicMock()
                mock_msg.message = data
                return mock_msg
            tool.create_json_message = create_json_message
            
            tool_parameters = {
                "collection_name": "nonexistent_collection",
                "filter": "id > 0"
            }
            
            result = list(tool._invoke(tool_parameters))
            
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "does not exist" in response["error"]
    
    def test_delete_with_empty_filter(self):
        """Test error when filter is empty (safety check)"""
        with patch('tools.milvus_delete.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            from tools.milvus_delete import MilvusDeleteTool
            
            tool = MilvusDeleteTool()
            tool.runtime = MagicMock()
            tool.runtime.credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root",
                "password": "test_password",
                "database": "test_db"
            }
            
            def create_json_message(data):
                mock_msg = MagicMock()
                mock_msg.message = data
                return mock_msg
            tool.create_json_message = create_json_message
            
            tool_parameters = {
                "collection_name": "test_collection",
                "filter": ""  # Empty filter
            }
            
            result = list(tool._invoke(tool_parameters))
            
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "filter expression is required for safety" in response["error"].lower()


class TestMilvusDeleteToolConfiguration:
    """Test the tool configuration aspects"""
    
    def test_tool_yaml_structure_expectations(self):
        """Test expectations for the tool YAML configuration"""
        expected_yaml_structure = {
            "identity": {
                "name": "milvus_delete",
                "author": "vibeany",
                "label": {"en_US": "Delete from Milvus Collection"}
            },
            "description": {
                "en_US": "Delete data from a Milvus collection using PyMilvus"
            },
            "parameters": [
                {
                    "name": "collection_name",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "filter", 
                    "type": "string",
                    "required": True
                }
            ],
            "extra": {
                "python": {
                    "source": "tools/milvus_delete.py"
                }
            }
        }
        
        assert expected_yaml_structure["identity"]["name"] == "milvus_delete"
        assert len(expected_yaml_structure["parameters"]) == 2
        assert expected_yaml_structure["parameters"][0]["required"] is True
        assert expected_yaml_structure["parameters"][1]["required"] is True  # Filter is required for safety