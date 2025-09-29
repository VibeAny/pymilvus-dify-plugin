"""
Milvus Collection List Tool

Lists all collections in the connected Milvus database using PyMilvus client.
This tool replaces HTTP API calls with pure PyMilvus gRPC operations.
"""
from typing import Any
from collections.abc import Generator
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .milvus_base import MilvusBaseTool

logger = logging.getLogger(__name__)


class MilvusCollectionListTool(Tool):
    """Tool for listing all collections in Milvus database"""
    
    def __init__(self, runtime=None, session=None):
        super().__init__(runtime, session)
        self.runtime = runtime
        self.session = session
        self.base_tool = MilvusBaseTool()
    
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        List all collections in the Milvus database
        
        Args:
            tool_parameters: Empty dict - no parameters needed for listing
            
        Returns:
            Generator yielding ToolInvokeMessage with collection list and count
        """
        logger.info("🚀 [DEBUG] MilvusCollectionListTool._invoke() called")
        
        try:
            logger.info("🔗 [DEBUG] Attempting to connect to Milvus...")
            
            with self.base_tool._get_milvus_client(self.runtime.credentials) as client:
                logger.info("✅ [DEBUG] Successfully connected to Milvus")
                
                # List all collections
                collections = client.list_collections()
                logger.info(f"📋 [DEBUG] Found {len(collections)} collections: {collections}")
                
                # Prepare response
                response = {
                    "success": True,
                    "collections": collections,
                    "count": len(collections)
                }
                
                logger.info(f"✅ [DEBUG] Operation completed successfully, result: {response}")
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
logger.info("📦 [DEBUG] milvus_collection_list.py module loaded")