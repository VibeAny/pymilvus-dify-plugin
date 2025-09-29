"""
Milvus Collection Describe Tool

Gets detailed information about a Milvus collection schema and configuration using PyMilvus client.
This tool replaces HTTP API calls with pure PyMilvus gRPC operations.
"""
from typing import Any
from collections.abc import Generator
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .milvus_base import MilvusBaseTool

logger = logging.getLogger(__name__)


class MilvusCollectionDescribeTool(Tool):
    """Tool for describing Milvus collection schema and configuration"""
    
    def __init__(self, runtime=None, session=None):
        super().__init__(runtime, session)
        self.runtime = runtime
        self.session = session
        self.base_tool = MilvusBaseTool()
    
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Describe a Milvus collection
        
        Args:
            tool_parameters: Contains collection_name parameter
            
        Returns:
            Generator yielding ToolInvokeMessage with collection description
        """
        logger.info(f"🚀 [DEBUG] MilvusCollectionDescribeTool._invoke() called with params: {tool_parameters}")
        
        try:
            collection_name = tool_parameters.get("collection_name")
            
            if not collection_name:
                raise ValueError("collection_name is required")
            
            logger.info("🔗 [DEBUG] Attempting to connect to Milvus...")
            
            with self.base_tool._get_milvus_client(self.runtime.credentials) as client:
                logger.info("✅ [DEBUG] Successfully connected to Milvus")
                
                # Check if collection exists
                if not client.has_collection(collection_name):
                    raise ValueError(f"Collection '{collection_name}' does not exist")
                
                # Get collection description
                description = client.describe_collection(collection_name)
                logger.info(f"📄 [DEBUG] Collection description retrieved: {description}")
                
                # Prepare response
                response = {
                    "success": True,
                    "collection_name": collection_name,
                    "description": description
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
logger.info("📦 [DEBUG] milvus_collection_describe.py module loaded")