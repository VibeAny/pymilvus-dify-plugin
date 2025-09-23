from typing import Any
import requests
import time

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError


class MilvusProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        """Validate Milvus connection and embedding provider configuration"""
        validation_errors = []
        
        try:
            # 1. Validate Milvus connection
            try:
                self._validate_milvus_connection(credentials)
            except ToolProviderCredentialValidationError as e:
                validation_errors.append(f"Milvus Connection: {str(e)}")
            
            # 2. Validate embedding provider configuration
            try:
                self._validate_embedding_provider(credentials)
            except ToolProviderCredentialValidationError as e:
                validation_errors.append(f"Embedding Provider: {str(e)}")
            
            # If there are any validation errors, raise combined error
            if validation_errors:
                error_message = " | ".join(validation_errors)
                raise ToolProviderCredentialValidationError(error_message)
                
        except ToolProviderCredentialValidationError:
            raise
        except Exception as e:
            raise ToolProviderCredentialValidationError(f"Configuration validation failed: {str(e)}")
    
    def _validate_milvus_connection(self, credentials: dict[str, Any]) -> None:
        """验证 Milvus 数据库连接"""
        # 获取连接参数
        uri = credentials.get("uri")
        token = credentials.get("token")
        database = credentials.get("database", "default")
        
        if not uri:
            raise ToolProviderCredentialValidationError("URI is required")
        
        # 确保 URI 格式正确
        if not uri.startswith(("http://", "https://")):
            uri = f"http://{uri}"
        
        # 移除末尾的斜杠
        uri = uri.rstrip('/')
        
        # 创建 HTTP 会话
        session = requests.Session()
        session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # 设置认证
        if token:
            session.headers['Authorization'] = f'Bearer {token}'
        
        # 测试连接 - 尝试列出集合
        test_url = f"{uri}/v2/vectordb/collections/list"
        
        try:
            response = session.post(test_url, json={}, timeout=10.0)
            
            # 检查 HTTP 状态码
            if response.status_code != 200:
                raise ToolProviderCredentialValidationError(
                    f"Failed to connect to Milvus server. HTTP {response.status_code}: {response.text}"
                )
            
            # 解析响应
            result = response.json()
            
            # 检查 Milvus 响应码
            if result.get('code') != 0:
                error_msg = result.get('message', 'Unknown error')
                raise ToolProviderCredentialValidationError(
                    f"Milvus API error: {error_msg}"
                )
            
        except requests.exceptions.RequestException as e:
            raise ToolProviderCredentialValidationError(
                f"Network error connecting to Milvus: {str(e)}"
            )
        finally:
            session.close()
    
    def _validate_embedding_provider(self, credentials: dict[str, Any]) -> None:
        """验证 Embedding 提供商配置"""
        embedding_provider = credentials.get("embedding_provider", "openai")
        
        # 只在用户实际配置了 API Key 时才验证
        if embedding_provider == "openai":
            api_key = credentials.get("openai_api_key")
            if api_key:  # 只有配置了 API Key 才验证
                self._validate_openai_credentials(credentials)
        elif embedding_provider == "azure_openai":
            api_key = credentials.get("azure_openai_api_key")
            endpoint = credentials.get("azure_openai_endpoint")
            if api_key and endpoint:  # 只有配置了完整信息才验证
                self._validate_azure_openai_credentials(credentials)
    
    def _validate_openai_credentials(self, credentials: dict[str, Any]) -> None:
        """验证 OpenAI API 配置"""
        api_key = credentials.get("openai_api_key")
        
        if not api_key:
            raise ToolProviderCredentialValidationError(
                "OpenAI API Key is required when using OpenAI provider"
            )
        
        # 测试 OpenAI API
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 使用最简单的模型列表 API 来验证
            response = requests.get(
                "https://api.openai.com/v1/models", 
                headers=headers, 
                timeout=10.0
            )
            
            if response.status_code != 200:
                raise ToolProviderCredentialValidationError(
                    f"OpenAI API Key validation failed - HTTP {response.status_code}: {response.text}"
                )
                
        except requests.exceptions.RequestException as e:
            raise ToolProviderCredentialValidationError(
                f"Failed to validate OpenAI API Key: {str(e)}"
            )
    
    def _validate_azure_openai_credentials(self, credentials: dict[str, Any]) -> None:
        """验证 Azure OpenAI API 配置"""
        endpoint = credentials.get("azure_openai_endpoint")
        api_key = credentials.get("azure_openai_api_key")
        api_version = credentials.get("azure_api_version", "2023-12-01-preview")
        
        if not endpoint:
            raise ToolProviderCredentialValidationError(
                "Azure OpenAI Endpoint is required when using Azure OpenAI provider"
            )
        
        if not api_key:
            raise ToolProviderCredentialValidationError(
                "Azure OpenAI API Key is required when using Azure OpenAI provider"
            )
        
        # 测试 Azure OpenAI API
        try:
            endpoint = endpoint.rstrip('/')
            headers = {
                "api-key": api_key,
                "Content-Type": "application/json"
            }
            
            # 使用 deployments 列表 API 来验证
            test_url = f"{endpoint}/openai/deployments"
            params = {"api-version": api_version}
            
            response = requests.get(
                test_url,
                headers=headers,
                params=params,
                timeout=10.0
            )
            
            if response.status_code != 200:
                raise ToolProviderCredentialValidationError(
                    f"Azure OpenAI API validation failed - HTTP {response.status_code}: {response.text}"
                )
                
        except requests.exceptions.RequestException as e:
            raise ToolProviderCredentialValidationError(
                f"Failed to validate Azure OpenAI API: {str(e)}"
            )
