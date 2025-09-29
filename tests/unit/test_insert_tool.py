"""
Unit tests for milvus_insert tool

Following TDD approach:
1. Red: Write failing tests first
2. Green: Write minimal code to make tests pass  
3. Refactor: Improve code while keeping tests green
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any


class TestMilvusInsertTool:
    """Test the milvus_insert tool using TDD"""
    
    def test_insert_tool_initialization(self):
        """Test that the tool can be initialized properly"""
        with patch('tools.milvus_insert.MilvusBaseTool'):
            from tools.milvus_insert import MilvusInsertTool
            
            tool = MilvusInsertTool()
            assert tool is not None
            assert hasattr(tool, '_invoke')
            assert hasattr(tool, 'base_tool')  # Composition pattern
    
    def test_insert_vector_data_success(self):
        """Test successful vector data insertion"""
        with patch('tools.milvus_insert.MilvusBaseTool') as mock_base_tool_class:
            # Mock the base tool instance
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            # Mock the client context manager
            mock_client = MagicMock()
            mock_client.has_collection.return_value = True
            mock_client.insert.return_value = {"insert_count": 2, "ids": [1, 2]}
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_insert import MilvusInsertTool
            
            tool = MilvusInsertTool()
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
                "data": [
                    {"vector": [0.1, 0.2, 0.3], "text": "first document"},
                    {"vector": [0.4, 0.5, 0.6], "text": "second document"}
                ]
            }
            
            # Execute the tool
            result = list(tool._invoke(tool_parameters))
            
            # Verify results
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is True
            assert response["collection_name"] == "test_collection"
            assert response["insert_count"] == 2
            assert response["ids"] == [1, 2]
            
            # Verify client was called correctly
            mock_client.has_collection.assert_called_once_with("test_collection")
            mock_client.insert.assert_called_once()
    
    def test_insert_missing_collection_name(self):
        """Test error when collection name is missing"""
        with patch('tools.milvus_insert.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            from tools.milvus_insert import MilvusInsertTool
            
            tool = MilvusInsertTool()
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
            
            tool_parameters = {"data": [{"vector": [0.1, 0.2, 0.3]}]}
            
            result = list(tool._invoke(tool_parameters))
            
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "collection_name is required" in response["error"].lower()
    
    def test_insert_empty_data(self):
        """Test error when data is empty"""
        with patch('tools.milvus_insert.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            from tools.milvus_insert import MilvusInsertTool
            
            tool = MilvusInsertTool()
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
                "data": []
            }
            
            result = list(tool._invoke(tool_parameters))
            
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "data is required and cannot be empty" in response["error"].lower()
    
    def test_insert_collection_not_exists(self):
        """Test error when collection does not exist"""
        with patch('tools.milvus_insert.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            mock_client = MagicMock()
            mock_client.has_collection.return_value = False
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_insert import MilvusInsertTool
            
            tool = MilvusInsertTool()
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
                "data": [{"vector": [0.1, 0.2, 0.3]}]
            }
            
            result = list(tool._invoke(tool_parameters))
            
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "does not exist" in response["error"]


class TestMilvusInsertToolConfiguration:
    """Test the tool configuration aspects"""
    
    def test_tool_yaml_structure_expectations(self):
        """Test expectations for the tool YAML configuration"""
        expected_yaml_structure = {
            "identity": {
                "name": "milvus_insert",
                "author": "vibeany",
                "label": {"en_US": "Insert Data into Milvus Collection"}
            },
            "description": {
                "en_US": "Insert vector data into a Milvus collection using PyMilvus"
            },
            "parameters": [
                {
                    "name": "collection_name",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "data", 
                    "type": "string",
                    "required": True
                }
            ],
            "extra": {
                "python": {
                    "source": "tools/milvus_insert.py"
                }
            }
        }
        
        assert expected_yaml_structure["identity"]["name"] == "milvus_insert"
        assert len(expected_yaml_structure["parameters"]) == 2
        assert expected_yaml_structure["parameters"][0]["required"] is True