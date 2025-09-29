"""
Milvus Query Tool

Queries data from a Milvus collection using PyMilvus client.
This tool replaces HTTP API calls with pure PyMilvus gRPC operations.
"""
from typing import Any
from collections.abc import Generator
import json
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .milvus_base import MilvusBaseTool

logger = logging.getLogger(__name__)


class MilvusQueryTool(Tool):
    """Tool for querying data from Milvus collections"""

    def __init__(self, runtime=None, session=None):
        super().__init__(runtime, session)
        self.runtime = runtime
        self.session = session
        self.base_tool = MilvusBaseTool()

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Query data from a Milvus collection
        
        Args:
            tool_parameters: Contains collection_name, filter, and query parameters
            
        Returns:
            Generator yielding ToolInvokeMessage with query results
        """
        logger.info(f"🚀 [DEBUG] MilvusQueryTool._invoke() called with params: {tool_parameters}")
        
        try:
            collection_name = tool_parameters.get("collection_name")
            filter_expr = tool_parameters.get("filter")
            limit = tool_parameters.get("limit", 100)
            
            if not collection_name:
                raise ValueError("collection_name is required")
            
            logger.info(f"🔍 [DEBUG] Querying collection: {collection_name}")
            logger.info("🔗 [DEBUG] Attempting to connect to Milvus...")
            
            with self.base_tool._get_milvus_client(self.runtime.credentials) as client:
                logger.info("✅ [DEBUG] Successfully connected to Milvus")
                
                # Check if collection exists
                if not client.has_collection(collection_name):
                    raise ValueError(f"Collection '{collection_name}' does not exist")
                
                # Perform query
                query_result = client.query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    limit=int(limit),
                    output_fields=tool_parameters.get("output_fields")
                )
                
                logger.info(f"✅ [DEBUG] Query completed: {len(query_result)} results found")
                
                # Prepare response
                response = {
                    "success": True,
                    "collection_name": collection_name,
                    "query_results": query_result,
                    "result_count": len(query_result)
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
logger.info("📦 [DEBUG] milvus_query.py module loaded")