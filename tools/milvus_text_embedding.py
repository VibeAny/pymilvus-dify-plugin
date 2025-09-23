import json
import logging
from typing import Any

from dify_plugin import Tool
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from pymilvus.model.dense import OpenAIEmbeddingFunction

logger = logging.getLogger(__name__)


class MilvusTextEmbeddingTool(Tool):
    def _invoke(self, user_id: str, tool_parameters: dict[str, Any]) -> str:
        """
        将文本转换为向量嵌入
        """
        try:
            # 获取参数
            text = tool_parameters.get("text", "").strip()
            model_name = tool_parameters.get("model", "text-embedding-3-small")
            normalize = tool_parameters.get("normalize", True)
            
            if not text:
                return json.dumps({
                    "success": False,
                    "error": "输入文本不能为空"
                }, ensure_ascii=False)
            
            # 获取向量嵌入
            embedding_result = self._get_text_embedding(text, model_name)
            
            if not embedding_result["success"]:
                return json.dumps(embedding_result, ensure_ascii=False)
            
            embedding_vector = embedding_result["embedding"]
            
            # 标准化向量（如果需要）
            if normalize:
                embedding_vector = self._normalize_vector(embedding_vector)
            
            result = {
                "success": True,
                "text": text,
                "embedding": embedding_vector,
                "dimension": len(embedding_vector),
                "model": model_name,
                "normalized": normalize,
                "provider": embedding_result.get("provider", "PyMilvus"),
                "message": f"成功将文本转换为 {len(embedding_vector)} 维向量"
            }
            
            logger.info(f"Text embedding successful: {len(embedding_vector)} dimensions")
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Text embedding failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": f"文本向量化失败: {str(e)}"
            }, ensure_ascii=False)
    
    def _get_text_embedding(self, text: str, model_name: str) -> dict:
        """
        使用 PyMilvus 获取文本向量（支持 OpenAI 和 Azure OpenAI）
        """
        try:
            credentials = self.runtime.credentials
            embedding_provider = credentials.get("embedding_provider", "openai")
            
            if embedding_provider == "openai":
                return self._get_openai_embedding(text, model_name, credentials)
            
            elif embedding_provider == "azure_openai":
                return self._get_azure_openai_embedding(text, model_name, credentials)
            
            else:
                return {
                    "success": False,
                    "error": f"不支持的嵌入提供商: {embedding_provider}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"文本向量化失败: {str(e)}"
            }
    
    def _get_openai_embedding(self, text: str, model_name: str, credentials: dict) -> dict:
        """
        使用 PyMilvus + OpenAI 获取向量
        """
        try:
            openai_key = credentials.get("openai_api_key")
            
            if not openai_key:
                return {
                    "success": False,
                    "error": "OpenAI API Key 未配置，请在插件设置中添加"
                }
            
            # 使用 PyMilvus OpenAI embedding 函数
            embedding_fn = OpenAIEmbeddingFunction(
                model_name=model_name,
                api_key=openai_key
            )
            
            # 获取向量
            query_vectors = embedding_fn.encode_queries([text])
            embedding_vector = query_vectors[0]
            
            # 转换为列表
            if hasattr(embedding_vector, 'tolist'):
                embedding_vector = embedding_vector.tolist()
            else:
                embedding_vector = list(embedding_vector)
            
            return {
                "success": True,
                "embedding": embedding_vector,
                "provider": "PyMilvus + OpenAI"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"PyMilvus OpenAI embedding 调用失败: {str(e)}"
            }
    
    def _get_azure_openai_embedding(self, text: str, model_name: str, credentials: dict) -> dict:
        """
        尝试使用 PyMilvus + Azure OpenAI，如果不支持则使用直接 API 调用
        """
        try:
            azure_endpoint = credentials.get("azure_openai_endpoint")
            azure_key = credentials.get("azure_openai_api_key")
            api_version = credentials.get("azure_api_version", "2023-12-01-preview")
            
            if not azure_endpoint or not azure_key:
                return {
                    "success": False,
                    "error": "Azure OpenAI 配置不完整，请检查 endpoint 和 API key"
                }
            
            # 尝试使用 PyMilvus 支持 Azure OpenAI（如果支持的话）
            try:
                # 构建 Azure OpenAI 的 base_url
                azure_base_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{model_name}"
                
                # 尝试创建支持自定义 base_url 的 embedding 函数
                embedding_fn = OpenAIEmbeddingFunction(
                    model_name=model_name,
                    api_key=azure_key,
                    base_url=azure_base_url  # 尝试传递 base_url
                )
                
                # 获取向量
                query_vectors = embedding_fn.encode_queries([text])
                embedding_vector = query_vectors[0]
                
                # 转换为列表
                if hasattr(embedding_vector, 'tolist'):
                    embedding_vector = embedding_vector.tolist()
                else:
                    embedding_vector = list(embedding_vector)
                
                return {
                    "success": True,
                    "embedding": embedding_vector,
                    "provider": "PyMilvus + Azure OpenAI"
                }
                
            except TypeError:
                # PyMilvus 不支持 base_url 参数，回退到直接 API 调用
                logger.warning("PyMilvus 不支持 Azure OpenAI base_url 参数，使用直接 API 调用")
                return self._call_azure_openai_direct(text, model_name, azure_endpoint, azure_key, api_version)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Azure OpenAI embedding 调用失败: {str(e)}"
            }
    
    def _call_azure_openai_direct(self, text: str, model_name: str, endpoint: str, api_key: str, api_version: str) -> dict:
        """
        直接调用 Azure OpenAI API（PyMilvus 不支持时的回退方案）
        """
        try:
            import requests
            
            endpoint = endpoint.rstrip('/')
            url = f"{endpoint}/openai/deployments/{model_name}/embeddings"
            
            headers = {
                "api-key": api_key,
                "Content-Type": "application/json"
            }
            
            payload = {"input": text}
            params = {"api-version": api_version}
            
            response = requests.post(url, headers=headers, json=payload, params=params, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                embedding = result["data"][0]["embedding"]
                return {
                    "success": True,
                    "embedding": embedding,
                    "provider": "Azure OpenAI (Direct API)"
                }
            else:
                return {
                    "success": False,
                    "error": f"Azure OpenAI API 错误: {response.status_code} - {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Azure OpenAI 直接调用失败: {str(e)}"
            }
    
    def _normalize_vector(self, vector: list) -> list:
        """
        标准化向量（L2 归一化）
        """
        import math
        
        norm = math.sqrt(sum(x * x for x in vector))
        
        if norm == 0:
            return vector
        
        return [x / norm for x in vector]