"""
Unit tests for milvus_text_embedding tool

Following TDD approach:
1. Red: Write failing tests first
2. Green: Write minimal code to make tests pass  
3. Refactor: Improve code while keeping tests green
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestMilvusTextEmbeddingTool:
    """Test the milvus_text_embedding tool using TDD"""
    
    def test_text_embedding_tool_initialization(self):
        """Test that the tool can be initialized properly"""
        from tools.milvus_text_embedding import MilvusTextEmbeddingTool
        
        tool = MilvusTextEmbeddingTool()
        assert tool is not None
        assert hasattr(tool, '_invoke')
        assert hasattr(tool, 'base_tool')
        assert hasattr(tool, '_get_text_embedding')
    
    def test_text_embedding_openai_success(self):
        """Test successful text embedding with OpenAI"""
        with patch('tools.milvus_text_embedding.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            mock_client = MagicMock()
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            with patch('tools.milvus_text_embedding.OpenAIEmbeddingFunction') as mock_openai:
                mock_embed_fn = MagicMock()
                mock_openai.return_value = mock_embed_fn
                mock_embed_fn.encode_queries.return_value = [[0.1] * 1536]
                
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
                
                tool.create_json_message = MagicMock(return_value="success_message")
                
                messages = list(tool._invoke({
                    "text": "Hello world",
                    "model": "text-embedding-3-small",
                    "normalize": False
                }))
                
                assert len(messages) == 1
                assert messages[0] == "success_message"
                tool.create_json_message.assert_called_once()
                call_args = tool.create_json_message.call_args[0][0]
                assert call_args["success"] is True
                assert "embedding" in call_args
                assert call_args["dimension"] == 1536
    
    def test_text_embedding_missing_text(self):
        """Test error handling when text is missing"""
        with patch('tools.milvus_text_embedding.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            mock_client = MagicMock()
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_text_embedding import MilvusTextEmbeddingTool
            
            tool = MilvusTextEmbeddingTool()
            tool.runtime = MagicMock()
            tool.runtime.credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root",
                "password": "test_password"
            }
            
            tool.create_json_message = MagicMock(return_value="error_message")
            
            messages = list(tool._invoke({"text": "", "model": "text-embedding-3-small"}))
            
            assert len(messages) == 1
            tool.create_json_message.assert_called_once()
            call_args = tool.create_json_message.call_args[0][0]
            assert call_args["success"] is False
            assert "error" in call_args
    
    def test_text_embedding_missing_api_key(self):
        """Test error handling when API key is missing"""
        with patch('tools.milvus_text_embedding.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            mock_client = MagicMock()
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            from tools.milvus_text_embedding import MilvusTextEmbeddingTool
            
            tool = MilvusTextEmbeddingTool()
            tool.runtime = MagicMock()
            tool.runtime.credentials = {
                "uri": "https://milvus-api.roomwits.com",
                "user": "root",
                "password": "test_password",
                "embedding_provider": "openai"
            }
            
            tool.create_json_message = MagicMock(return_value="error_message")
            
            messages = list(tool._invoke({"text": "Hello world", "model": "text-embedding-3-small"}))
            
            assert len(messages) == 1
            tool.create_json_message.assert_called_once()
            call_args = tool.create_json_message.call_args[0][0]
            assert call_args["success"] is False
            assert "error" in call_args
    
    def test_text_embedding_azure_openai_success(self):
        """Test successful text embedding with Azure OpenAI"""
        with patch('tools.milvus_text_embedding.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            mock_client = MagicMock()
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            with patch('tools.milvus_text_embedding.OpenAIEmbeddingFunction') as mock_openai:
                mock_embed_fn = MagicMock()
                mock_openai.return_value = mock_embed_fn
                mock_embed_fn.encode_queries.return_value = [[0.2] * 1536]
                
                from tools.milvus_text_embedding import MilvusTextEmbeddingTool
                
                tool = MilvusTextEmbeddingTool()
                tool.runtime = MagicMock()
                tool.runtime.credentials = {
                    "uri": "https://milvus-api.roomwits.com",
                    "user": "root",
                    "password": "test_password",
                    "embedding_provider": "azure_openai",
                    "azure_openai_endpoint": "https://test.openai.azure.com",
                    "azure_openai_api_key": "test_azure_key",
                    "azure_api_version": "2023-12-01-preview"
                }
                
                tool.create_json_message = MagicMock(return_value="success_message")
                
                messages = list(tool._invoke({
                    "text": "Azure test",
                    "model": "text-embedding-ada-002",
                    "normalize": False
                }))
                
                assert len(messages) == 1
                assert messages[0] == "success_message"
    
    def test_text_embedding_normalization(self):
        """Test vector normalization"""
        with patch('tools.milvus_text_embedding.MilvusBaseTool') as mock_base_tool_class:
            mock_base_tool_instance = MagicMock()
            mock_base_tool_class.return_value = mock_base_tool_instance
            
            mock_client = MagicMock()
            mock_base_tool_instance._get_milvus_client.return_value.__enter__.return_value = mock_client
            
            with patch('tools.milvus_text_embedding.OpenAIEmbeddingFunction') as mock_openai:
                mock_embed_fn = MagicMock()
                mock_openai.return_value = mock_embed_fn
                mock_embed_fn.encode_queries.return_value = [[3.0, 4.0]]
                
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
                
                tool.create_json_message = MagicMock(return_value="success_message")
                
                messages = list(tool._invoke({
                    "text": "Normalize test",
                    "model": "text-embedding-3-small",
                    "normalize": True
                }))
                
                assert len(messages) == 1
                call_args = tool.create_json_message.call_args[0][0]
                assert call_args["success"] is True
                assert call_args["normalized"] is True


class TestMilvusTextEmbeddingToolConfiguration:
    """Test tool YAML configuration structure expectations"""
    
    def test_tool_yaml_structure_expectations(self):
        """Test that the YAML file has expected structure for text embedding tool"""
        import yaml
        from pathlib import Path
        
        yaml_path = Path(__file__).parent.parent.parent / "tools" / "milvus_text_embedding.yaml"
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert "identity" in config
            assert config["identity"]["name"] == "milvus_text_embedding"
            assert "parameters" in config
            
            param_names = [p["name"] for p in config["parameters"]]
            assert "text" in param_names
            assert "model" in param_names