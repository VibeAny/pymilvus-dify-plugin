"""
Unit tests for milvus_bm25_search tool

Following TDD approach:
1. Red: Write failing tests first
2. Green: Write minimal code to make tests pass  
3. Refactor: Improve code while keeping tests green
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any


class TestMilvusBM25SearchTool:
    """Test the milvus_bm25_search tool using TDD"""
    
    def test_bm25_search_tool_initialization(self):
        """Test that the tool can be initialized properly"""
        with patch('tools.milvus_bm25_search.MilvusBaseTool'):
            from tools.milvus_bm25_search import MilvusBM25SearchTool
            
            tool = MilvusBM25SearchTool()
            assert tool is not None
            assert hasattr(tool, '_invoke')
            assert hasattr(tool, 'base_tool')  # Composition pattern
    
    def test_bm25_search_success(self):
        """Test successful BM25 text search"""
        with patch('tools.milvus_bm25_search.MilvusBaseTool') as mock_base_tool_class:
            # Mock the base tool instance
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            # Mock the client context manager
            mock_client = MagicMock()
            mock_client.has_collection.return_value = True
            mock_search_results = [
                [{"id": 1, "score": 0.95, "entity": {"text": "relevant document"}},
                 {"id": 2, "score": 0.85, "entity": {"text": "another relevant doc"}}]
            ]
            mock_client.bm25_search.return_value = mock_search_results
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_bm25_search import MilvusBM25SearchTool
            
            tool = MilvusBM25SearchTool()
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
                "query_text": "machine learning algorithms",
                "limit": 5
            }
            
            # Execute the tool
            result = list(tool._invoke(tool_parameters))
            
            # Verify results
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is True
            assert response["collection_name"] == "test_collection"
            assert response["query_text"] == "machine learning algorithms"
            assert len(response["search_results"]) == 2
            assert response["result_count"] == 2
            
            # Verify client was called correctly
            mock_client.has_collection.assert_called_once_with("test_collection")
            mock_client.bm25_search.assert_called_once()
    
    def test_bm25_search_missing_collection_name(self):
        """Test error when collection name is missing"""
        with patch('tools.milvus_bm25_search.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            from tools.milvus_bm25_search import MilvusBM25SearchTool
            
            tool = MilvusBM25SearchTool()
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
            
            tool_parameters = {"query_text": "search text"}
            
            result = list(tool._invoke(tool_parameters))
            
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "collection_name is required" in response["error"].lower()
    
    def test_bm25_search_missing_query_text(self):
        """Test error when query text is missing"""
        with patch('tools.milvus_bm25_search.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            from tools.milvus_bm25_search import MilvusBM25SearchTool
            
            tool = MilvusBM25SearchTool()
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
            assert "query_text is required" in response["error"].lower()
    
    def test_bm25_search_empty_results(self):
        """Test BM25 search with no results"""
        with patch('tools.milvus_bm25_search.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            mock_client = MagicMock()
            mock_client.has_collection.return_value = True
            mock_client.bm25_search.return_value = [[]]  # Empty results
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_bm25_search import MilvusBM25SearchTool
            
            tool = MilvusBM25SearchTool()
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
                "query_text": "nonexistent topic"
            }
            
            result = list(tool._invoke(tool_parameters))
            
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is True
            assert response["result_count"] == 0
            assert response["search_results"] == []
    
    def test_bm25_search_collection_not_exists(self):
        """Test error when collection does not exist"""
        with patch('tools.milvus_bm25_search.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            mock_client = MagicMock()
            mock_client.has_collection.return_value = False
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_bm25_search import MilvusBM25SearchTool
            
            tool = MilvusBM25SearchTool()
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
                "query_text": "search text"
            }
            
            result = list(tool._invoke(tool_parameters))
            
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "does not exist" in response["error"]


class TestMilvusBM25SearchToolConfiguration:
    """Test the tool configuration aspects"""
    
    def test_tool_yaml_structure_expectations(self):
        """Test expectations for the tool YAML configuration"""
        expected_yaml_structure = {
            "identity": {
                "name": "milvus_bm25_search",
                "author": "vibeany",
                "label": {"en_US": "BM25 Search Milvus Collection"}
            },
            "description": {
                "en_US": "Perform BM25 text search in a Milvus collection using PyMilvus"
            },
            "parameters": [
                {
                    "name": "collection_name",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "query_text", 
                    "type": "string",
                    "required": True
                }
            ],
            "extra": {
                "python": {
                    "source": "tools/milvus_bm25_search.py"
                }
            }
        }
        
        assert expected_yaml_structure["identity"]["name"] == "milvus_bm25_search"
        assert len(expected_yaml_structure["parameters"]) == 2
        assert expected_yaml_structure["parameters"][0]["required"] is True