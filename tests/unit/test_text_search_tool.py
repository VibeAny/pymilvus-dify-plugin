"""
Unit tests for milvus_text_search tool

Following TDD approach:
1. Red: Write failing tests first
2. Green: Write minimal code to make tests pass  
3. Refactor: Improve code while keeping tests green
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any


class TestMilvusTextSearchTool:
    """Test the milvus_text_search tool using TDD"""
    
    def test_text_search_tool_initialization(self):
        """Test that the tool can be initialized properly"""
        with patch('tools.milvus_text_search.MilvusBaseTool'):
            from tools.milvus_text_search import MilvusTextSearchTool
            
            tool = MilvusTextSearchTool()
            assert tool is not None
            assert hasattr(tool, '_invoke')
            assert hasattr(tool, '_get_text_embedding')
    
    def test_text_search_success(self):
        """Test successful text search (embedding + vector search)"""
        with patch('tools.milvus_text_search.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            mock_client = MagicMock()
            mock_client.search.return_value = [[
                {"id": 1, "distance": 0.95, "entity": {"text": "result 1"}},
                {"id": 2, "distance": 0.85, "entity": {"text": "result 2"}}
            ]]
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            with patch('tools.milvus_text_search.OpenAIEmbeddingFunction') as mock_openai:
                mock_embed_fn = MagicMock()
                mock_openai.return_value = mock_embed_fn
                mock_embed_fn.encode_queries.return_value = [[0.1] * 1536]
                
                from tools.milvus_text_search import MilvusTextSearchTool
                
                tool = MilvusTextSearchTool()
                tool.runtime = MagicMock()
                tool.runtime.credentials = {
                    "uri": "https://milvus-api.roomwits.com",
                    "user": "root",
                    "password": "test_password",
                    "embedding_provider": "openai",
                    "openai_api_key": "test_key"
                }
                
                tool.create_json_message = MagicMock(return_value="success_message")
                tool._validate_collection_name = MagicMock(return_value=True)
                
                messages = list(tool._invoke({
                    "collection_name": "test_collection",
                    "query_text": "Hello world",
                    "limit": 5,
                    "embedding_model": "text-embedding-3-small"
                }))
                
                assert len(messages) == 1
                tool.create_json_message.assert_called_once()
                call_args = tool.create_json_message.call_args[0][0]
                assert call_args["success"] is True
                assert "results" in call_args
    
    def test_text_search_missing_collection_name(self):
        """Test error handling when collection name is missing"""
        with patch('tools.milvus_text_search.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            from tools.milvus_text_search import MilvusTextSearchTool
            
            tool = MilvusTextSearchTool()
            tool.runtime = MagicMock()
            tool.runtime.credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root",
                "password": "test_password"
            }
            
            tool.create_json_message = MagicMock(return_value="error_message")
            
            messages = list(tool._invoke({
                "query_text": "Hello world"
            }))
            
            assert len(messages) == 1
            call_args = tool.create_json_message.call_args[0][0]
            assert call_args["success"] is False
            assert "error" in call_args
    
    def test_text_search_missing_query_text(self):
        """Test error handling when query text is missing"""
        with patch('tools.milvus_text_search.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            from tools.milvus_text_search import MilvusTextSearchTool
            
            tool = MilvusTextSearchTool()
            tool.runtime = MagicMock()
            tool.runtime.credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root",
                "password": "test_password"
            }
            
            tool.create_json_message = MagicMock(return_value="error_message")
            
            messages = list(tool._invoke({
                "collection_name": "test_collection",
                "query_text": ""
            }))
            
            assert len(messages) == 1
            call_args = tool.create_json_message.call_args[0][0]
            assert call_args["success"] is False
            assert "error" in call_args
    
    def test_text_search_embedding_failure(self):
        """Test error handling when text embedding fails"""
        with patch('tools.milvus_text_search.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            mock_client = MagicMock()
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            with patch('tools.milvus_text_search.OpenAIEmbeddingFunction') as mock_openai:
                mock_openai.side_effect = Exception("API Error")
                
                from tools.milvus_text_search import MilvusTextSearchTool
                
                tool = MilvusTextSearchTool()
                tool.runtime = MagicMock()
                tool.runtime.credentials = {
                    "uri": "https://milvus-api.roomwits.com",
                    "user": "root",
                    "password": "test_password",
                    "embedding_provider": "openai",
                    "openai_api_key": "test_key"
                }
                
                tool.create_json_message = MagicMock(return_value="error_message")
                tool._validate_collection_name = MagicMock(return_value=True)
                
                messages = list(tool._invoke({
                    "collection_name": "test_collection",
                    "query_text": "Hello world"
                }))
                
                assert len(messages) == 1
                call_args = tool.create_json_message.call_args[0][0]
                assert call_args["success"] is False
    
    def test_text_search_with_filter(self):
        """Test text search with filter expression"""
        with patch('tools.milvus_text_search.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            mock_client = MagicMock()
            mock_client.search.return_value = [[
                {"id": 1, "distance": 0.95, "entity": {"text": "filtered result"}}
            ]]
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            with patch('tools.milvus_text_search.OpenAIEmbeddingFunction') as mock_openai:
                mock_embed_fn = MagicMock()
                mock_openai.return_value = mock_embed_fn
                mock_embed_fn.encode_queries.return_value = [[0.1] * 1536]
                
                from tools.milvus_text_search import MilvusTextSearchTool
                
                tool = MilvusTextSearchTool()
                tool.runtime = MagicMock()
                tool.runtime.credentials = {
                    "uri": "https://milvus-api.roomwits.com",
                    "user": "root",
                    "password": "test_password",
                    "embedding_provider": "openai",
                    "openai_api_key": "test_key"
                }
                
                tool.create_json_message = MagicMock(return_value="success_message")
                tool._validate_collection_name = MagicMock(return_value=True)
                
                messages = list(tool._invoke({
                    "collection_name": "test_collection",
                    "query_text": "Hello world",
                    "limit": 5,
                    "filter": "category == 'tech'"
                }))
                
                assert len(messages) == 1
                call_args = tool.create_json_message.call_args[0][0]
                assert call_args["success"] is True
    
    def test_text_search_min_similarity_filter(self):
        """Test text search with minimum similarity filtering"""
        with patch('tools.milvus_text_search.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            mock_client = MagicMock()
            mock_client.search.return_value = [[
                {"id": 1, "distance": 0.95, "entity": {"text": "high similarity"}},
                {"id": 2, "distance": 0.5, "entity": {"text": "low similarity"}}
            ]]
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            with patch('tools.milvus_text_search.OpenAIEmbeddingFunction') as mock_openai:
                mock_embed_fn = MagicMock()
                mock_openai.return_value = mock_embed_fn
                mock_embed_fn.encode_queries.return_value = [[0.1] * 1536]
                
                from tools.milvus_text_search import MilvusTextSearchTool
                
                tool = MilvusTextSearchTool()
                tool.runtime = MagicMock()
                tool.runtime.credentials = {
                    "uri": "https://milvus-api.roomwits.com",
                    "user": "root",
                    "password": "test_password",
                    "embedding_provider": "openai",
                    "openai_api_key": "test_key"
                }
                
                tool.create_json_message = MagicMock(return_value="success_message")
                tool._validate_collection_name = MagicMock(return_value=True)
                
                messages = list(tool._invoke({
                    "collection_name": "test_collection",
                    "query_text": "Hello world",
                    "limit": 10,
                    "min_similarity": 0.8
                }))
                
                assert len(messages) == 1
                call_args = tool.create_json_message.call_args[0][0]
                assert call_args["success"] is True


class TestMilvusTextSearchToolConfiguration:
    """Test tool YAML configuration structure expectations"""
    
    def test_tool_yaml_structure_expectations(self):
        """Test that the YAML file has expected structure for text search tool"""
        import yaml
        from pathlib import Path
        
        yaml_path = Path(__file__).parent.parent.parent / "tools" / "milvus_text_search.yaml"
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert "identity" in config
            assert config["identity"]["name"] == "milvus_text_search"
            assert "parameters" in config
            
            param_names = [p["name"] for p in config["parameters"]]
            assert "collection_name" in param_names
            assert "query_text" in param_names