import json
import logging
from typing import Any

from dify_plugin import Tool
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from pymilvus.model.dense import OpenAIEmbeddingFunction
import requests

logger = logging.getLogger(__name__)


class MilvusTextSearchTool(Tool):
    def _invoke(self, user_id: str, tool_parameters: dict[str, Any]) -> str:
        """
        基于文本查询执行语义搜索（自动向量化+搜索）
        """
        try:
            # 获取参数
            collection_name = tool_parameters.get("collection_name")
            query_text = tool_parameters.get("query_text", "").strip()
            limit = int(tool_parameters.get("limit", 5))
            output_fields = tool_parameters.get("output_fields", "")
            filter_expr = tool_parameters.get("filter", "")
            embedding_model = tool_parameters.get("embedding_model", "text-embedding-3-small")
            anns_field = tool_parameters.get("anns_field", "vector")
            metric_type = tool_parameters.get("metric_type", "COSINE")
            min_similarity = tool_parameters.get("min_similarity")
            
            if not collection_name:
                return json.dumps({
                    "success": False,
                    "error": "集合名称不能为空"
                }, ensure_ascii=False)
            
            if not query_text:
                return json.dumps({
                    "success": False,
                    "error": "查询文本不能为空"
                }, ensure_ascii=False)
            
            # 获取认证信息
            credentials = self.runtime.credentials
            uri = credentials.get("uri")
            token = credentials.get("token")
            
            if not uri:
                return json.dumps({
                    "success": False,
                    "error": "Milvus URI 未配置"
                }, ensure_ascii=False)
            
            # Step 1: 将查询文本转换为向量
            logger.info(f"Converting query text to vector: {query_text[:100]}...")
            embedding_result = self._get_text_embedding(query_text, embedding_model)
            
            if not embedding_result["success"]:
                return json.dumps({
                    "success": False,
                    "error": f"文本向量化失败: {embedding_result['error']}"
                }, ensure_ascii=False)
            
            query_vector = embedding_result["embedding"]
            
            # Step 2: 执行向量搜索
            logger.info(f"Performing vector search in collection: {collection_name}")
            search_result = self._perform_vector_search(
                uri, token, collection_name, query_vector, limit, 
                output_fields, filter_expr, anns_field, metric_type
            )
            
            if not search_result["success"]:
                return json.dumps(search_result, ensure_ascii=False)
            
            # Step 3: 处理搜索结果
            results = search_result["results"]
            
            # 应用最小相似度过滤
            if min_similarity is not None:
                min_similarity = float(min_similarity)
                filtered_results = []
                for result in results:
                    score = result.get("score", 0)
                    # 对于 COSINE 距离，分数越接近 1 越相似
                    if metric_type == "L2":
                        similarity = 1 / (1 + score)  # 简单转换
                    else:
                        similarity = score
                    
                    if similarity >= min_similarity:
                        result["similarity"] = similarity
                        filtered_results.append(result)
                
                results = filtered_results
            
            result = {
                "success": True,
                "query_text": query_text,
                "collection_name": collection_name,
                "embedding_model": embedding_model,
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
                "message": f"在集合 {collection_name} 中找到 {len(results)} 个相关结果"
            }
            
            logger.info(f"Text search completed: {len(results)} results found")
            return json.dumps(result, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"Text search failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": f"文本搜索失败: {str(e)}"
            }, ensure_ascii=False)
    
    def _get_text_embedding(self, text: str, model_name: str) -> dict:
        """
        使用 PyMilvus 获取文本向量
        """
        try:
            credentials = self.runtime.credentials
            embedding_provider = credentials.get("embedding_provider", "openai")
            
            if embedding_provider == "openai":
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
                "error": f"PyMilvus embedding 调用失败: {str(e)}"
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
    
    def _perform_vector_search(self, uri: str, token: str, collection_name: str, 
                             query_vector: list, limit: int, output_fields: str,
                             filter_expr: str, anns_field: str, metric_type: str) -> dict:
        """
        执行向量搜索
        """
        try:
            # 确保 URI 格式正确
            if not uri.startswith(("http://", "https://")):
                uri = f"http://{uri}"
            
            uri = uri.rstrip('/')
            
            # 构建搜索请求
            search_url = f"{uri}/v2/vectordb/entities/search"
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # 设置认证
            if token:
                headers['Authorization'] = f'Bearer {token}'
            
            # 构建搜索参数
            search_data = {
                "collectionName": collection_name,
                "data": [query_vector],
                "annsField": anns_field,
                "limit": limit,
                "searchParams": {
                    "metric_type": metric_type,
                    "params": {"level": 1}
                }
            }
            
            # 添加输出字段
            if output_fields:
                fields = [field.strip() for field in output_fields.split(',')]
                search_data["outputFields"] = fields
            
            # 添加过滤条件
            if filter_expr:
                search_data["filter"] = filter_expr
            
            # 执行搜索请求
            response = requests.post(search_url, json=search_data, headers=headers, timeout=30)
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Milvus 搜索 API 错误: {response.status_code} - {response.text}"
                }
            
            result = response.json()
            
            # 检查 Milvus 响应
            if result.get('code') != 0:
                return {
                    "success": False,
                    "error": f"Milvus 搜索失败: {result.get('message', 'Unknown error')}"
                }
            
            # 解析搜索结果
            search_results = []
            if result.get('data') and len(result['data']) > 0:
                for hit in result['data'][0]:  # 第一个查询的结果
                    result_item = {
                        "id": hit.get("id"),
                        "score": hit.get("distance", hit.get("score", 0)),
                        "entity": hit.get("entity", {})
                    }
                    search_results.append(result_item)
            
            return {
                "success": True,
                "results": search_results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"向量搜索执行失败: {str(e)}"
            }