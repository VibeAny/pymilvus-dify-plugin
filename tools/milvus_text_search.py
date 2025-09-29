"""
Milvus Text Search Tool using Dify's Built-in Model System

This tool performs semantic text search by combining Dify's text embedding
capabilities with Milvus vector search, eliminating the need for users to
configure embedding provider credentials.
"""
from typing import Any
from collections.abc import Generator
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .milvus_base import MilvusBaseTool

logger = logging.getLogger(__name__)


class MilvusTextSearchTool(Tool):
    """Tool for semantic text search using Dify's model system + Milvus vector search"""
    
    def __init__(self, runtime=None, session=None):
        super().__init__(runtime, session)
        self.runtime = runtime
        self.session = session
        self.base_tool = MilvusBaseTool()
    
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Perform semantic search using text query (automatic embedding + search)
        
        Args:
            tool_parameters: Contains search parameters and text query
            
        Returns:
            Generator yielding ToolInvokeMessage with search results
        """
        logger.info(f"🚀 [DEBUG] MilvusTextSearchTool._invoke() called with params: {tool_parameters}")
        
        try:
            collection_name = tool_parameters.get("collection_name")
            query_text = tool_parameters.get("query_text", "").strip()
            model_config = tool_parameters.get("model")
            limit = int(tool_parameters.get("limit", 5))
            output_fields = tool_parameters.get("output_fields", "")
            filter_expr = tool_parameters.get("filter", "")
            anns_field = tool_parameters.get("anns_field", "vector")
            metric_type = tool_parameters.get("metric_type", "COSINE")
            min_similarity = tool_parameters.get("min_similarity")
            
            # Debug logging for model parameter
            logger.info(f"🔧 [DEBUG] Received model parameter: {model_config} (type: {type(model_config)})")
            logger.info(f"🔧 [DEBUG] All tool parameters: {list(tool_parameters.keys())}")
            
            if not collection_name:
                raise ValueError("Collection name is required")
            
            if not query_text:
                raise ValueError("Query text is required")
            
            if not model_config:
                raise ValueError("Model configuration is required (select an embedding model)")
            
            if not self.base_tool._validate_collection_name(collection_name):
                raise ValueError("Invalid collection name format")
            
            logger.info(f"🔍 [DEBUG] Performing text search in collection: {collection_name}")
            
            # Step 1: Get text embedding using Dify's model system
            embedding_result = self._get_dify_embedding(
                text=query_text,
                model_config=model_config
            )
            
            if not embedding_result["success"]:
                raise ValueError(f"Text embedding failed: {embedding_result['error']}")
            
            query_vector = embedding_result["embedding"]
            
            logger.info(f"📊 [DEBUG] Generated embedding vector, dimension: {len(query_vector)}")
            
            # Step 2: Perform vector search
            with self.base_tool._get_milvus_client(self.runtime.credentials) as client:
                output_field_list = []
                if output_fields:
                    output_field_list = [field.strip() for field in output_fields.split(',')]
                
                search_params = {}
                if metric_type:
                    search_params["metric_type"] = metric_type
                
                logger.info(f"🔍 [DEBUG] Performing vector search with limit: {limit}")
                
                search_results = client.search(
                    collection_name=collection_name,
                    data=[query_vector],
                    anns_field=anns_field,
                    limit=limit,
                    output_fields=output_field_list if output_field_list else None,
                    filter=filter_expr if filter_expr else None,
                    search_params=search_params if search_params else None
                )
                
                # Process search results
                results = []
                if search_results and len(search_results) > 0:
                    for hit in search_results[0]:
                        score = hit.get("distance", hit.get("score", 0))
                        
                        # Convert distance to similarity based on metric type
                        if metric_type == "L2":
                            similarity = 1 / (1 + score)
                        else:
                            similarity = score
                        
                        # Apply similarity filtering if specified
                        if min_similarity is not None and similarity < float(min_similarity):
                            continue
                        
                        result_item = {
                            "id": hit.get("id"),
                            "score": score,
                            "similarity": similarity,
                            "entity": hit.get("entity", {})
                        }
                        results.append(result_item)
                
                # Build result data
                result_data = {
                    "status": "success",
                    "operation": "text_search",
                    "query_text": query_text,
                    "collection_name": collection_name,
                    "model_provider": embedding_result.get("provider", "unknown"),
                    "model_name": embedding_result.get("model", "unknown"),
                    "vector_dimension": len(query_vector),
                    "total_results": len(results),
                    "results": results,
                    "search_params": {
                        "limit": limit,
                        "metric_type": metric_type,
                        "anns_field": anns_field,
                        "filter": filter_expr,
                        "min_similarity": min_similarity
                    },
                    "usage": embedding_result.get("usage", {}),
                    "message": f"Found {len(results)} relevant results for query '{query_text}' in collection {collection_name}"
                }
                
                logger.info(f"✅ [DEBUG] Text search completed successfully, found {len(results)} results")
                
                yield self.create_json_message(result_data)
                
        except Exception as e:
            logger.error(f"❌ [DEBUG] Error in text search: {type(e).__name__}: {str(e)}", exc_info=True)
            
            error_data = {
                "status": "error",
                "operation": "text_search",
                "error": str(e),
                "error_type": type(e).__name__,
                "message": f"Text search failed: {str(e)}"
            }
            
            yield self.create_json_message(error_data)
    
    def _get_dify_embedding(self, text: str, model_config) -> dict:
        """
        Get text embedding using Dify's built-in model system
        
        Args:
            text: Input text to embed
            model_config: Model configuration object from model-selector parameter
            
        Returns:
            Dict with success status and embedding data
        """
        try:
            logger.info(f"🔍 [DEBUG] Invoking embedding model: {model_config}")
            
            # Convert dict model_config to proper TextEmbeddingModelConfig object
            from dify_plugin.entities.model.text_embedding import TextEmbeddingModelConfig
            
            if isinstance(model_config, dict):
                # Extract provider and model from dict
                provider = model_config.get('provider', '').split('/')[-1]  # Remove prefix like 'langgenius/azure_openai/'
                model = model_config.get('model')
                
                # Create proper TextEmbeddingModelConfig object
                model_config_obj = TextEmbeddingModelConfig(
                    provider=provider,
                    model=model
                )
            else:
                # Already a proper object
                model_config_obj = model_config
            
            # Use Dify's text_embedding API with proper model_config object
            embedding_result = self.session.model.text_embedding.invoke(
                model_config=model_config_obj,
                texts=[text]
            )
            
            if not embedding_result:
                return {
                    "success": False,
                    "error": "Embedding model returned empty result"
                }
            
            # Extract embedding vector from result
            if hasattr(embedding_result, 'embeddings') and embedding_result.embeddings:
                embedding_vector = embedding_result.embeddings[0]
            elif isinstance(embedding_result, list) and embedding_result:
                embedding_vector = embedding_result[0]
            else:
                return {
                    "success": False,
                    "error": "Unable to extract embedding vector from model result"
                }
            
            # Convert to list if needed
            if hasattr(embedding_vector, 'tolist'):
                embedding_vector = embedding_vector.tolist()
            elif not isinstance(embedding_vector, list):
                embedding_vector = list(embedding_vector)
            
            # Extract usage information if available
            usage_info = {}
            if hasattr(embedding_result, 'usage'):
                usage_info = {
                    "tokens": getattr(embedding_result.usage, 'total_tokens', 0),
                    "prompt_tokens": getattr(embedding_result.usage, 'prompt_tokens', 0)
                }
            
            logger.info(f"✅ [DEBUG] Embedding generated successfully, dimension: {len(embedding_vector)}")
            
            return {
                "success": True,
                "embedding": embedding_vector,
                "usage": usage_info,
                "provider": f"Dify ({getattr(model_config_obj, 'provider', 'unknown')})",
                "model": getattr(model_config_obj, 'model', 'unknown')
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ [DEBUG] Dify embedding failed: {error_msg}", exc_info=True)
            
            # Provide more specific error handling (same as text embedding tool)
            if "permission denied" in error_msg.lower():
                return {
                    "success": False,
                    "error": "Permission denied: Text embedding access not enabled. Please check plugin manifest permissions or contact administrator.",
                    "error_type": "PermissionError",
                    "suggestion": "Ensure 'text_embedding: true' is set in manifest.yaml under resource.permission.model"
                }
            elif "handshake failed" in error_msg.lower() or "invalid key" in error_msg.lower():
                return {
                    "success": False,
                    "error": "Connection authentication failed. Plugin key may be expired or invalid.",
                    "error_type": "AuthenticationError",
                    "suggestion": "Please restart the plugin or check Dify server connection"
                }
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                return {
                    "success": False,
                    "error": "Network connection issue with Dify server. Please check connectivity.",
                    "error_type": "ConnectionError",
                    "suggestion": "Verify Dify server is running and accessible"
                }
            elif "model not found" in error_msg.lower() or "model_config" in error_msg.lower():
                return {
                    "success": False,
                    "error": "Selected embedding model is not available or not configured in Dify workspace.",
                    "error_type": "ModelError", 
                    "suggestion": "Please select a different embedding model or configure the model in Dify"
                }
            else:
                return {
                    "success": False,
                    "error": f"Dify model system embedding failed: {error_msg}",
                    "error_type": "UnknownError",
                    "suggestion": "Please check logs for more details"
                }
    
    def _handle_error(self, error: Exception, operation: str = "text_search") -> dict:
        """
        Format error for consistent error handling
        
        Args:
            error: Exception that occurred
            operation: Operation name for context
            
        Returns:
            Formatted error dictionary
        """
        error_message = str(error)
        error_type = type(error).__name__
        
        logger.error(f"❌ [DEBUG] {operation} error: {error_type}: {error_message}")
        
        return {
            "status": "error",
            "operation": operation,
            "error": error_message,
            "error_type": error_type,
            "message": f"Operation {operation} failed: {error_message}"
        }