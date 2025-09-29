"""
Milvus Delete Tool

Deletes data from a Milvus collection using PyMilvus client.
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


class MilvusDeleteTool(Tool):
    """Tool for deleting data from Milvus collections"""

    def __init__(self, runtime=None, session=None):
        super().__init__(runtime, session)
        self.runtime = runtime
        self.session = session
        self.base_tool = MilvusBaseTool()

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Delete data from a Milvus collection
        
        Args:
            tool_parameters: Contains collection_name and filter parameters
            
        Returns:
            Generator yielding ToolInvokeMessage with deletion result
        """
        logger.info(f"🚀 [DEBUG] MilvusDeleteTool._invoke() called with params: {tool_parameters}")
        
        try:
            collection_name = tool_parameters.get("collection_name")
            filter_expr = tool_parameters.get("filter")
            
            if not collection_name:
                raise ValueError("collection_name is required")
            
            if not filter_expr:
                raise ValueError("filter expression is required for safety - cannot delete all data")
            
            logger.info(f"🗑️ [DEBUG] Deleting from collection: {collection_name}")
            logger.info("🔗 [DEBUG] Attempting to connect to Milvus...")
            
            with self.base_tool._get_milvus_client(self.runtime.credentials) as client:
                logger.info("✅ [DEBUG] Successfully connected to Milvus")
                
                # Check if collection exists
                if not client.has_collection(collection_name):
                    raise ValueError(f"Collection '{collection_name}' does not exist")
                
                # Perform deletion
                delete_result = client.delete(
                    collection_name=collection_name,
                    filter=filter_expr
                )
                
                logger.info(f"✅ [DEBUG] Deletion completed successfully")
                
                # Prepare response
                response = {
                    "success": True,
                    "collection_name": collection_name,
                    "filter": filter_expr,
                    "delete_count": delete_result.get("delete_count", 0) if delete_result else 0
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
logger.info("📦 [DEBUG] milvus_delete.py module loaded")