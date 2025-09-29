"""
Unit tests for milvus_text_embedding and milvus_text_search tools

Testing fallback mechanisms when PyMilvus model is not available
"""
import pytest
from unittest.mock import patch, MagicMock


class TestMilvusTextEmbeddingTool:
    """Test the milvus_text_embedding tool"""
    
    def test_text_embedding_tool_initialization(self):
        """Test that the tool can be initialized"""
        from tools.milvus_text_embedding import MilvusTextEmbeddingTool
        
        tool = MilvusTextEmbeddingTool()
        assert tool is not None
        assert hasattr(tool, 'base_tool')
        assert hasattr(tool, '_invoke')
    
    def test_text_embedding_fallback_to_direct_api(self):
        """Test that embedding falls back to direct OpenAI API when PyMilvus model not available"""
        with patch('tools.milvus_text_embedding.HAS_PYMILVUS_MODEL', False):
            with patch('requests.post') as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "data": [{"embedding": [0.1] * 1536}]
                }
                mock_post.return_value = mock_response
                
                from tools.milvus_text_embedding import MilvusTextEmbeddingTool
                
                tool = MilvusTextEmbeddingTool()
                tool.runtime = MagicMock()
                tool.runtime.credentials = {
                    "uri": "https://milvus-api.roomwits.com",
                    "user": "root",
                    "password": "test_password",
                    "embedding_provider": "openai",
                    "openai_api_key": "test_key"
                }
                
                with patch.object(tool.base_tool, '_get_milvus_client') as mock_client_ctx:
                    mock_client = MagicMock()
                    mock_client_ctx.return_value.__enter__.return_value = mock_client
                    
                    messages = list(tool._invoke({
                        "text": "Hello world",
                        "model": "text-embedding-3-small",
                        "normalize": False
                    }))
                    
                    assert len(messages) == 1


class TestMilvusTextSearchTool:
    """Test the milvus_text_search tool"""
    
    def test_text_search_tool_initialization(self):
        """Test that the tool can be initialized"""
        from tools.milvus_text_search import MilvusTextSearchTool
        
        tool = MilvusTextSearchTool()
        assert tool is not None
        assert hasattr(tool, 'base_tool')
        assert hasattr(tool, '_invoke')
    
    def test_text_search_fallback_to_direct_api(self):
        """Test that search falls back to direct OpenAI API when PyMilvus model not available"""
        with patch('tools.milvus_text_search.HAS_PYMILVUS_MODEL', False):
            with patch('requests.post') as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "data": [{"embedding": [0.1] * 1536}]
                }
                mock_post.return_value = mock_response
                
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
                
                with patch.object(tool.base_tool, '_get_milvus_client') as mock_client_ctx:
                    with patch.object(tool.base_tool, '_validate_collection_name', return_value=True):
                        mock_client = MagicMock()
                        mock_client.search.return_value = [[
                            {"id": 1, "distance": 0.95, "entity": {"text": "result 1"}}
                        ]]
                        mock_client_ctx.return_value.__enter__.return_value = mock_client
                        
                        messages = list(tool._invoke({
                            "collection_name": "test_collection",
                            "query_text": "Hello world",
                            "limit": 5
                        }))
                        
                        assert len(messages) == 1