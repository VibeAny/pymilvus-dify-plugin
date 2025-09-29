"""
Simplified integration tests for all Milvus tools

Testing that tools can be instantiated and have required methods
"""
import pytest


class TestAllMilvusToolsIntegration:
    """Test all Milvus tools can be imported and initialized"""
    
    def test_collection_tools_can_be_imported(self):
        """Test that collection management tools can be imported"""
        from tools.milvus_collection_list import MilvusCollectionListTool
        from tools.milvus_collection_describe import MilvusCollectionDescribeTool
        from tools.milvus_collection_create import MilvusCollectionCreateTool
        from tools.milvus_collection_drop import MilvusCollectionDropTool
        
        # Test initialization
        list_tool = MilvusCollectionListTool()
        describe_tool = MilvusCollectionDescribeTool()
        create_tool = MilvusCollectionCreateTool()
        drop_tool = MilvusCollectionDropTool()
        
        # Test that all have required attributes
        for tool in [list_tool, describe_tool, create_tool, drop_tool]:
            assert hasattr(tool, 'base_tool')
            assert hasattr(tool, '_invoke')
    
    def test_data_operation_tools_can_be_imported(self):
        """Test that data operation tools can be imported"""
        from tools.milvus_insert import MilvusInsertTool
        from tools.milvus_query import MilvusQueryTool
        from tools.milvus_search import MilvusSearchTool
        from tools.milvus_delete import MilvusDeleteTool
        
        # Test initialization
        insert_tool = MilvusInsertTool()
        query_tool = MilvusQueryTool()
        search_tool = MilvusSearchTool()
        delete_tool = MilvusDeleteTool()
        
        # Test that all have required attributes
        for tool in [insert_tool, query_tool, search_tool, delete_tool]:
            assert hasattr(tool, 'base_tool')
            assert hasattr(tool, '_invoke')
    
    def test_search_tools_can_be_imported(self):
        """Test that search tools can be imported"""
        from tools.milvus_bm25_search import MilvusBM25SearchTool
        from tools.milvus_text_embedding import MilvusTextEmbeddingTool
        from tools.milvus_text_search import MilvusTextSearchTool
        
        # Test initialization
        bm25_tool = MilvusBM25SearchTool()
        embedding_tool = MilvusTextEmbeddingTool()
        text_search_tool = MilvusTextSearchTool()
        
        # Test that all have required attributes
        for tool in [bm25_tool, embedding_tool, text_search_tool]:
            assert hasattr(tool, 'base_tool')
            assert hasattr(tool, '_invoke')
    
    def test_base_tool_functionality(self):
        """Test that base tool has core functionality"""
        from tools.milvus_base import MilvusBaseTool
        
        base_tool = MilvusBaseTool()
        
        # Test core methods exist
        assert hasattr(base_tool, '_get_milvus_client')
        assert hasattr(base_tool, '_validate_collection_name')
        assert hasattr(base_tool, '_parse_vector_data')
        assert callable(base_tool._get_milvus_client)
        assert callable(base_tool._validate_collection_name)
        assert callable(base_tool._parse_vector_data)
    
    def test_all_tools_use_composition_pattern(self):
        """Test that all tools use the composition pattern correctly"""
        from tools.milvus_collection_list import MilvusCollectionListTool
        from tools.milvus_insert import MilvusInsertTool
        from tools.milvus_search import MilvusSearchTool
        from tools.milvus_bm25_search import MilvusBM25SearchTool
        from tools.milvus_text_embedding import MilvusTextEmbeddingTool
        
        tools = [
            MilvusCollectionListTool(),
            MilvusInsertTool(),
            MilvusSearchTool(),
            MilvusBM25SearchTool(),
            MilvusTextEmbeddingTool()
        ]
        
        # All should have base_tool attribute and it should be MilvusBaseTool
        for tool in tools:
            assert hasattr(tool, 'base_tool')
            assert tool.base_tool is not None
            # Check that base_tool has the expected methods
            assert hasattr(tool.base_tool, '_get_milvus_client')
            assert hasattr(tool.base_tool, '_validate_collection_name')


class TestMilvusProviderConfiguration:
    """Test provider configuration structure"""
    
    def test_provider_yaml_lists_all_tools(self):
        """Test that provider.yaml includes all our tools"""
        import yaml
        from pathlib import Path
        
        yaml_path = Path(__file__).parent.parent.parent / "provider" / "milvus.yaml"
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "tools" in config
        tool_list = config["tools"]
        
        # Check that all our new tools are listed
        expected_tools = [
            "tools/milvus_collection_create.yaml",
            "tools/milvus_collection_list.yaml",
            "tools/milvus_collection_describe.yaml", 
            "tools/milvus_collection_drop.yaml",
            "tools/milvus_search.yaml",
            "tools/milvus_insert.yaml",
            "tools/milvus_query.yaml",
            "tools/milvus_delete.yaml",
            "tools/milvus_text_embedding.yaml",
            "tools/milvus_text_search.yaml",
            "tools/milvus_bm25_search.yaml"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_list, f"Tool {expected_tool} not found in provider configuration"