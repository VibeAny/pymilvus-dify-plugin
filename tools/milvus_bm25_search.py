"""
Milvus BM25 Search Tool

Performs BM25 sparse vector search in a Milvus collection using PyMilvus client.
This tool replaces HTTP API calls with pure PyMilvus gRPC operations.
"""
from typing import Any
from collections.abc import Generator
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .milvus_base import MilvusBaseTool

logger = logging.getLogger(__name__)


class MilvusBM25SearchTool(Tool):
    """Tool for BM25 text search in Milvus collections"""

    def __init__(self, runtime=None, session=None):
        super().__init__(runtime, session)
        self.runtime = runtime
        self.session = session
        self.base_tool = MilvusBaseTool()

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Perform BM25 keyword search
        
        Args:
            tool_parameters: Contains collection_name, query_text, and search parameters
            
        Returns:
            Generator yielding ToolInvokeMessage with BM25 search results
        """
        logger.info(f"🚀 [DEBUG] MilvusBM25SearchTool._invoke() called with params: {tool_parameters}")
        
        try:
            collection_name = tool_parameters.get("collection_name")
            query_text = tool_parameters.get("query_text")
            limit = tool_parameters.get("limit", 10)
            
            if not collection_name:
                raise ValueError("collection_name is required")
            
            if not query_text:
                raise ValueError("query_text is required")
            
            logger.info(f"🔍 [DEBUG] BM25 search in collection: {collection_name}")
            logger.info("🔗 [DEBUG] Attempting to connect to Milvus...")
            
            with self.base_tool._get_milvus_client(self.runtime.credentials) as client:
                logger.info("✅ [DEBUG] Successfully connected to Milvus")
                
                # Check if collection exists
                if not client.has_collection(collection_name):
                    raise ValueError(f"Collection '{collection_name}' does not exist")
                
                # Perform BM25 search
                search_result = client.bm25_search(
                    collection_name=collection_name,
                    query_text=query_text,
                    limit=int(limit),
                    output_fields=tool_parameters.get("output_fields")
                )
                
                logger.info(f"✅ [DEBUG] BM25 search completed: {len(search_result)} results found")
                
                # Prepare response
                response = {
                    "success": True,
                    "collection_name": collection_name,
                    "query_text": query_text,
                    "search_results": search_result[0] if search_result else [],
                    "result_count": len(search_result[0]) if search_result else 0
                }
                
                logger.info(f"✅ [DEBUG] Operation completed successfully")
                yield self.create_json_message(response)
                
        except Exception as e:
            logger.error(f"❌ [DEBUG] Error in _invoke(): {type(e).__name__}: {str(e)}", exc_info=True)
            yield from self._handle_error(e)
    
    def _handle_error(self, error: Exception) -> Generator[ToolInvokeMessage]:
        """Handle and format errors consistently"""
        logger.error(f"🚨 [DEBUG] _handle_error() called with: {type(error).__name__}: {str(error)}")
        
        error_msg = str(error)
        response = {
            "success": False,
            "error": error_msg,
            "error_type": type(error).__name__
        }
        
        logger.debug(f"📤 [DEBUG] Sending error response: {response}")
        yield self.create_json_message(response)


# Module level debug info
logger.info("📦 [DEBUG] milvus_bm25_search.py module loaded")