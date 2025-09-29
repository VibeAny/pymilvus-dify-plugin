"""
Refactored Milvus base tool class using PyMilvus client wrapper

This replaces the HTTP API approach with pure PyMilvus SDK integration.
Provides common functionality for all Milvus tools with proper error handling.
"""
from typing import Any, Optional, Dict, List
from collections.abc import Generator
from contextlib import contextmanager
import json
import logging

logger = logging.getLogger(__name__)

# Import the PyMilvus client wrapper
from lib.milvus_client import MilvusClientWrapper


class MilvusBaseTool:
    """Milvus 工具基类，提供通用的 PyMilvus 连接和错误处理功能"""
    
    @contextmanager
    def _get_milvus_client(self, credentials: dict[str, Any]):
        """创建 PyMilvus 客户端的上下文管理器"""
        try:
            # Validate required credentials
            uri = credentials.get("uri")
            user = credentials.get("user")
            password = credentials.get("password")
            database = credentials.get("database", "default")
            
            if not uri:
                raise ValueError("URI is required")
            if not user:
                raise ValueError("Username is required")
            if not password:
                raise ValueError("Password is required")
            
            # Ensure complete credentials for client wrapper
            complete_credentials = {
                "uri": uri,
                "user": user,
                "password": password,
                "database": database
            }
            
            # Create PyMilvus client wrapper
            client = MilvusClientWrapper(complete_credentials)
            
            logger.info(f"✅ [DEBUG] Successfully connected to Milvus gRPC API at {uri}")
            yield client
            
        except Exception as e:
            logger.error(f"❌ [DEBUG] Failed to connect to Milvus: {str(e)}")
            raise ValueError(f"Failed to connect to Milvus: {str(e)}")
    
    def _validate_collection_name(self, collection_name: str) -> bool:
        """验证集合名称"""
        if not collection_name or not isinstance(collection_name, str):
            return False
        if len(collection_name) > 255:
            return False
        # 集合名称只能包含字母、数字和下划线，且不能以数字开头
        import re
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', collection_name))
    
    def _parse_vector_data(self, data: Any) -> list:
        """解析向量数据"""
        try:
            if isinstance(data, str):
                return json.loads(data)
            elif isinstance(data, list):
                return data
            else:
                raise ValueError("Data must be string or list")
        except (json.JSONDecodeError, TypeError, ValueError):
            raise ValueError("Invalid vector data format. Expected JSON array.")
    
    def _parse_search_params(self, params_str: Optional[str]) -> dict:
        """解析搜索参数"""
        if not params_str:
            return {}
        
        try:
            return json.loads(params_str)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def _format_error_message(self, error: Exception, operation: Optional[str] = None) -> str:
        """格式化错误消息"""
        error_str = str(error)
        if operation:
            return f"Error during {operation}: {error_str}"
        return f"Milvus operation failed: {error_str}"
    
    def _create_error_response(self, error: Exception, operation: str = "operation") -> dict:
        """创建标准化的错误响应"""
        error_msg = str(error)
        error_type = type(error).__name__
        
        # Provide specific error handling based on error content
        if "permission denied" in error_msg.lower():
            return {
                "success": False,
                "error": "Permission denied: Required access not enabled in plugin configuration",
                "error_type": "PermissionError",
                "operation": operation,
                "suggestion": "Please check plugin manifest permissions or contact administrator"
            }
        elif "handshake failed" in error_msg.lower() or "invalid key" in error_msg.lower():
            return {
                "success": False,
                "error": "Authentication failed: Plugin connection key is invalid or expired",
                "error_type": "AuthenticationError", 
                "operation": operation,
                "suggestion": "Please restart the plugin or check Dify server connection"
            }
        elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            return {
                "success": False,
                "error": "Network connectivity issue: Cannot reach required service",
                "error_type": "ConnectionError",
                "operation": operation,
                "suggestion": "Verify network connectivity and service availability"
            }
        elif "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
            return {
                "success": False,
                "error": f"Resource not found: {error_msg}",
                "error_type": "ResourceNotFoundError",
                "operation": operation,
                "suggestion": "Please verify the resource name and ensure it exists"
            }
        else:
            return {
                "success": False,
                "error": f"{operation.title()} failed: {error_msg}",
                "error_type": error_type,
                "operation": operation,
                "suggestion": "Please check the input parameters and try again"
            }