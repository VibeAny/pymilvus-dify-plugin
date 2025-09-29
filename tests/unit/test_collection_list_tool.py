"""
Unit tests for milvus_collection_list tool

Following TDD approach:
1. Red: Write failing tests first
2. Green: Write minimal code to make tests pass  
3. Refactor: Improve code while keeping tests green
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any


class TestMilvusCollectionListTool:
    """Test the milvus_collection_list tool using TDD"""
    
    def test_collection_list_tool_initialization(self):
        """Test that the tool can be initialized properly"""
        with patch('tools.milvus_collection_list.MilvusBaseTool'):
            from tools.milvus_collection_list import MilvusCollectionListTool
            
            tool = MilvusCollectionListTool()
            assert tool is not None
            assert hasattr(tool, '_invoke')
            assert hasattr(tool, 'base_tool')  # Composition pattern
    
    def test_collection_list_success(self):
        """Test successful collection listing"""
        with patch('tools.milvus_collection_list.MilvusBaseTool') as mock_base_tool_class:
            # Mock the base tool instance
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            # Mock the client context manager
            mock_client = MagicMock()
            mock_client.list_collections.return_value = ["collection1", "collection2", "test_collection"]
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_collection_list import MilvusCollectionListTool
            
            tool = MilvusCollectionListTool()
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
            
            tool_parameters = {}
            
            # Execute the tool
            result = list(tool._invoke(tool_parameters))
            
            # Verify results
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is True
            assert "collections" in response
            assert response["collections"] == ["collection1", "collection2", "test_collection"]
            assert response["count"] == 3
            
            # Verify client was called correctly
            mock_client.list_collections.assert_called_once()
    
    def test_collection_list_empty_result(self):
        """Test collection listing with no collections"""
        with patch('tools.milvus_collection_list.MilvusBaseTool') as mock_base_tool_class:
            # Mock the base tool instance
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            # Mock the client context manager
            mock_client = MagicMock()
            mock_client.list_collections.return_value = []
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_collection_list import MilvusCollectionListTool
            
            tool = MilvusCollectionListTool()
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
            
            tool_parameters = {}
            
            # Execute the tool
            result = list(tool._invoke(tool_parameters))
            
            # Verify results
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is True
            assert "collections" in response
            assert response["collections"] == []
            assert response["count"] == 0
    
    def test_collection_list_connection_error(self):
        """Test handling of connection errors"""
        with patch('tools.milvus_collection_list.MilvusBaseTool') as mock_base_tool_class:
            # Mock the base tool instance
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            # Mock the client context manager to raise exception
            mock_base_tool_instance._get_milvus_client.side_effect = Exception("Connection failed")
            
            from tools.milvus_collection_list import MilvusCollectionListTool
            
            tool = MilvusCollectionListTool()
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
            
            tool_parameters = {}
            
            # Execute the tool
            result = list(tool._invoke(tool_parameters))
            
            # Verify error response
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "error" in response
            assert "Connection failed" in response["error"]
    
    def test_collection_list_milvus_client_error(self):
        """Test handling of Milvus client errors"""
        with patch('tools.milvus_collection_list.MilvusBaseTool') as mock_base_tool_class:
            # Mock the base tool instance
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            # Mock the client context manager
            mock_client = MagicMock()
            mock_client.list_collections.side_effect = Exception("Milvus API error")
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_collection_list import MilvusCollectionListTool
            
            tool = MilvusCollectionListTool()
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
            
            tool_parameters = {}
            
            # Execute the tool
            result = list(tool._invoke(tool_parameters))
            
            # Verify error response
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "error" in response
            assert "Milvus API error" in response["error"]
    
    def test_collection_list_invalid_credentials(self):
        """Test handling of invalid credentials"""
        with patch('tools.milvus_collection_list.MilvusBaseTool') as mock_base_tool_class:
            # Mock the base tool instance
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            # Mock the client context manager to raise credential error
            mock_base_tool_instance._get_milvus_client.side_effect = ValueError("URI is required")
            
            from tools.milvus_collection_list import MilvusCollectionListTool
            
            tool = MilvusCollectionListTool()
            tool.runtime = MagicMock()
            tool.runtime.credentials = {
                "uri": "",  # Missing URI
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
            
            tool_parameters = {}
            
            # Execute the tool
            result = list(tool._invoke(tool_parameters))
            
            # Verify error response
            assert len(result) == 1
            response = result[0].message
            assert response["success"] is False
            assert "error" in response
            assert "URI is required" in response["error"]


class TestMilvusCollectionListToolConfiguration:
    """Test the tool configuration aspects"""
    
    def test_tool_yaml_structure_expectations(self):
        """Test expectations for the tool YAML configuration"""
        # This test validates what we expect the YAML to contain
        expected_yaml_structure = {
            "identity": {
                "name": "milvus_collection_list",
                "author": "vibeany",
                "label": {"en_US": "List Milvus Collections"}
            },
            "description": {
                "en_US": "List all collections in Milvus database using PyMilvus"
            },
            "parameters": [],  # No parameters needed for listing
            "extra": {
                "python": {
                    "source": "tools/milvus_collection_list.py"
                }
            }
        }
        
        # Verify the structure is sensible
        assert expected_yaml_structure["identity"]["name"] == "milvus_collection_list"
        assert expected_yaml_structure["parameters"] == []  # List operation needs no parameters