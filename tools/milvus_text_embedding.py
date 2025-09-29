"""
Milvus Text Embedding Tool using Dify's Built-in Model System

This tool converts text to vector embeddings using Dify's model provider system,
which allows using any configured embedding model in Dify without direct API calls.
"""
from typing import Any
from collections.abc import Generator
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from .milvus_base import MilvusBaseTool

logger = logging.getLogger(__name__)


class MilvusTextEmbeddingTool(Tool):
    """Tool for converting text to embeddings using Dify's model system"""
    
    def __init__(self, runtime=None, session=None):
        super().__init__(runtime, session)
        self.runtime = runtime
        self.session = session
        self.base_tool = MilvusBaseTool()
    
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Convert text to vector embedding using Dify's built-in model system
        
        Args:
            tool_parameters: Contains text and optional model configuration
            
        Returns:
            Generator yielding ToolInvokeMessage with embedding result
        """
        logger.info(f"🚀 [DEBUG] MilvusTextEmbeddingTool._invoke() called with params: {tool_parameters}")
        
        try:
            text = tool_parameters.get("text", "").strip()
            model_config = tool_parameters.get("model")
            normalize = tool_parameters.get("normalize", True)
            
            if not text:
                raise ValueError("Input text is required for embedding")
            
            if not model_config:
                raise ValueError("Model configuration is required (select an embedding model)")
            
            logger.info(f"📝 [DEBUG] Processing text embedding with model: {model_config}")
            
            # Use Dify's model system to get embedding
            embedding_result = self._get_dify_embedding(
                text=text,
                model_config=model_config
            )
            
            if not embedding_result["success"]:
                raise ValueError(f"Text embedding failed: {embedding_result['error']}")
            
            embedding_vector = embedding_result["embedding"]
            
            # Optional vector normalization
            if normalize:
                embedding_vector = self._normalize_vector(embedding_vector)
            
            # Build result data
            result_data = {
                "status": "success",
                "operation": "text_embedding",
                "text": text,
                "embedding": embedding_vector,
                "dimension": len(embedding_vector),
                "model_provider": getattr(model_config, 'provider', 'unknown'),
                "model_name": getattr(model_config, 'model', 'unknown'),
                "normalized": normalize,
                "usage": embedding_result.get("usage", {}),
                "message": f"Successfully converted text to {len(embedding_vector)}-dimensional vector using Dify model system"
            }
            
            logger.info(f"✅ [DEBUG] Text embedding completed successfully, dimension: {len(embedding_vector)}")
            
            yield self.create_json_message(result_data)
            
        except Exception as e:
            logger.error(f"❌ [DEBUG] Error in text embedding: {type(e).__name__}: {str(e)}", exc_info=True)
            
            # Extract detailed error information if available
            if hasattr(e, '__cause__') and e.__cause__ and "embedding_result" in str(e):
                # This is likely an error from _get_dify_embedding
                embedding_result = {"success": False, "error": str(e)}
            else:
                # Direct exception
                embedding_result = {
                    "success": False, 
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            
            # Create user-friendly error response
            error_data = {
                "status": "error",
                "operation": "text_embedding",
                "error": embedding_result.get("error", str(e)),
                "error_type": embedding_result.get("error_type", type(e).__name__),
                "message": f"Text embedding failed: {embedding_result.get('error', str(e))}",
                "suggestion": embedding_result.get("suggestion", "Please check the input parameters and try again")
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
                "provider": f"Dify ({getattr(model_config, 'provider', 'unknown')})",
                "model": getattr(model_config, 'model', 'unknown')
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ [DEBUG] Dify embedding failed: {error_msg}", exc_info=True)
            
            # Provide more specific error handling
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
    
    def _normalize_vector(self, vector: list) -> list:
        """
        Normalize vector using L2 normalization
        
        Args:
            vector: Input vector to normalize
            
        Returns:
            Normalized vector
        """
        import math
        
        try:
            # Calculate L2 norm
            norm = math.sqrt(sum(x * x for x in vector))
            
            if norm == 0:
                logger.warning("⚠️ [DEBUG] Vector has zero norm, returning original vector")
                return vector
            
            # Normalize
            normalized = [x / norm for x in vector]
            logger.debug(f"🔢 [DEBUG] Vector normalized, original norm: {norm:.6f}")
            
            return normalized
            
        except Exception as e:
            logger.error(f"❌ [DEBUG] Vector normalization failed: {str(e)}")
            return vector  # Return original on error
    
    def _handle_error(self, error: Exception, operation: str = "text_embedding") -> dict:
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