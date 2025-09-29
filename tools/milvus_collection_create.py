"""
Milvus Collection Create Tool

Creates a new Milvus collection with specified schema using PyMilvus client.
This tool replaces HTTP API calls with pure PyMilvus gRPC operations.
"""
from typing import Any
from collections.abc import Generator
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .milvus_base import MilvusBaseTool

logger = logging.getLogger(__name__)


class MilvusCollectionCreateTool(Tool):
    """Tool for creating new Milvus collections with schema definition"""
    
    def __init__(self, runtime=None, session=None):
        super().__init__(runtime, session)
        self.runtime = runtime
        self.session = session
        self.base_tool = MilvusBaseTool()
    
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Create a new Milvus collection
        
        Args:
            tool_parameters: Contains collection_name, dimension, and optional parameters
            
        Returns:
            Generator yielding ToolInvokeMessage with creation result
        """
        logger.info(f"🚀 [DEBUG] MilvusCollectionCreateTool._invoke() called with params: {tool_parameters}")
        
        try:
            collection_name = tool_parameters.get("collection_name")
            dimension = tool_parameters.get("dimension")
            
            if not collection_name:
                raise ValueError("collection_name is required")
            
            if not dimension:
                raise ValueError("dimension is required")
            
            try:
                dimension = int(dimension)
            except (ValueError, TypeError):
                raise ValueError("dimension must be a valid integer")
            
            if dimension <= 0 or dimension > 32768:
                raise ValueError("dimension must be between 1 and 32768")
            
            # Get optional parameters
            metric_type = tool_parameters.get("metric_type", "COSINE")
            auto_id = tool_parameters.get("auto_id", True)
            description = tool_parameters.get("description", "")
            enable_bm25 = tool_parameters.get("enable_bm25", False)
            
            logger.info(f"🆕 [DEBUG] Creating collection: {collection_name}, dim: {dimension}, metric: {metric_type}")
            
            logger.info("🔗 [DEBUG] Attempting to connect to Milvus...")
            
            with self.base_tool._get_milvus_client(self.runtime.credentials) as client:
                logger.info("✅ [DEBUG] Successfully connected to Milvus")
                
                # Check if collection already exists
                if client.has_collection(collection_name):
                    raise ValueError(f"Collection '{collection_name}' already exists")
                
                # Create collection with schema
                schema_config = {
                    "collection_name": collection_name,
                    "enable_bm25": enable_bm25,
                    "vector_field": {
                        "name": "vector",
                        "dim": dimension
                    }
                }
                
                # Add text field if BM25 is enabled
                if enable_bm25:
                    schema_config["text_field"] = {
                        "name": "text",
                        "max_length": 65535
                    }
                
                client.create_collection_with_schema(schema_config)
                
                logger.info("✅ [DEBUG] Collection created successfully")
                
                # Prepare response
                response = {
                    "success": True,
                    "collection_name": collection_name,
                    "dimension": dimension,
                    "metric_type": metric_type,
                    "auto_id": auto_id,
                    "description": description,
                    "enable_bm25": enable_bm25
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
logger.info("📦 [DEBUG] milvus_collection_create.py module loaded")