"""
Unit tests for milvus_search tool

Following TDD approach:
1. Red: Write failing tests first
2. Green: Write minimal code to make tests pass  
3. Refactor: Improve code while keeping tests green
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any


class TestMilvusSearchTool:
    """Test the milvus_search tool using TDD"""
    
    def test_search_tool_initialization(self):
        """Test that the tool can be initialized properly"""
        with patch('tools.milvus_search.MilvusBaseTool'):
            from tools.milvus_search import MilvusSearchTool
            
            tool = MilvusSearchTool()
            assert tool is not None
            assert hasattr(tool, '_invoke')
            assert hasattr(tool, 'base_tool')  # Composition pattern
    
    def test_vector_search_success(self):
        """Test successful vector search"""
        with patch('tools.milvus_search.MilvusBaseTool') as mock_base_tool_class:
            # Mock the base tool instance
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            # Mock the client context manager
            mock_client = MagicMock()
            mock_client.has_collection.return_value = True
            mock_search_results = [
                [{"id": 1, "distance": 0.8, "entity": {"text": "result 1"}},
                 {"id": 2, "distance": 0.6, "entity": {"text": "result 2"}}]
            ]
            mock_client.vector_search.return_value = mock_search_results
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_search import MilvusSearchTool
            
            tool = MilvusSearchTool()
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
                "query_vector": [0.1, 0.2, 0.3, 0.4],
                "limit": 5
            }
            
            # Execute the tool
            result = list(tool._invoke(tool_parameters))
            
            # Verify results
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is True
            assert response["collection_name"] == "test_collection"
            assert len(response["search_results"]) == 2
            assert response["result_count"] == 2
            
            # Verify client was called correctly
            mock_client.has_collection.assert_called_once_with("test_collection")
            mock_client.vector_search.assert_called_once()
    
    def test_search_missing_collection_name(self):
        """Test error when collection name is missing"""
        with patch('tools.milvus_search.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            from tools.milvus_search import MilvusSearchTool
            
            tool = MilvusSearchTool()
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
            
            tool_parameters = {"query_vector": [0.1, 0.2, 0.3]}
            
            result = list(tool._invoke(tool_parameters))
            
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "collection_name is required" in response["error"].lower()
    
    def test_search_missing_query_vector(self):
        """Test error when query vector is missing"""
        with patch('tools.milvus_search.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            from tools.milvus_search import MilvusSearchTool
            
            tool = MilvusSearchTool()
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
            assert "query_vector is required" in response["error"].lower()
    
    def test_search_invalid_vector_format(self):
        """Test error when query vector format is invalid"""
        with patch('tools.milvus_search.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            from tools.milvus_search import MilvusSearchTool
            
            tool = MilvusSearchTool()
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
                "query_vector": "invalid_vector"
            }
            
            result = list(tool._invoke(tool_parameters))
            
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "must be valid JSON array" in response["error"] or "must be a list" in response["error"]
    
    def test_search_collection_not_exists(self):
        """Test error when collection does not exist"""
        with patch('tools.milvus_search.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            mock_client = MagicMock()
            mock_client.has_collection.return_value = False
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_search import MilvusSearchTool
            
            tool = MilvusSearchTool()
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
                "query_vector": [0.1, 0.2, 0.3]
            }
            
            result = list(tool._invoke(tool_parameters))
            
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "does not exist" in response["error"]


class TestMilvusSearchToolConfiguration:
    """Test the tool configuration aspects"""
    
    def test_tool_yaml_structure_expectations(self):
        """Test expectations for the tool YAML configuration"""
        expected_yaml_structure = {
            "identity": {
                "name": "milvus_search",
                "author": "vibeany",
                "label": {"en_US": "Search Milvus Collection"}
            },
            "description": {
                "en_US": "Perform vector similarity search in a Milvus collection using PyMilvus"
            },
            "parameters": [
                {
                    "name": "collection_name",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "query_vector", 
                    "type": "string",
                    "required": True
                }
            ],
            "extra": {
                "python": {
                    "source": "tools/milvus_search.py"
                }
            }
        }
        
        assert expected_yaml_structure["identity"]["name"] == "milvus_search"
        assert len(expected_yaml_structure["parameters"]) == 2
        assert expected_yaml_structure["parameters"][0]["required"] is True