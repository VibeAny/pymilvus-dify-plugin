from typing import Any
from collections.abc import Generator
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from pymilvus import MilvusClient
from .milvus_base import MilvusBaseTool

logger = logging.getLogger(__name__)


class MilvusBM25SearchTool(MilvusBaseTool, Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        使用BM25算法执行关键词文本搜索
        """
        logger.info(f"🚀 [DEBUG] MilvusBM25SearchTool._invoke() called with params: {tool_parameters}")
        
        try:
            # 获取参数
            collection_name = tool_parameters.get("collection_name")
            query_text = tool_parameters.get("query_text", "").strip()
            limit = int(tool_parameters.get("limit", 5))
            output_fields = tool_parameters.get("output_fields", "")
            text_field = tool_parameters.get("text_field", "document_content")
            filter_expr = tool_parameters.get("filter", "")
            bm25_k1 = float(tool_parameters.get("bm25_k1", 1.2))
            bm25_b = float(tool_parameters.get("bm25_b", 0.75))
            
            logger.debug(f"📋 [DEBUG] BM25 Search - Collection: {collection_name}, Query: {query_text}")
            
            if not collection_name:
                raise ValueError("Collection name is required")
            
            if not query_text:
                raise ValueError("Query text is required")
            
            if not self._validate_collection_name(collection_name):
                raise ValueError("Invalid collection name format")
            
            logger.info("🔗 [DEBUG] Attempting to connect to Milvus for BM25 search...")
            
            # 使用 MilvusBaseTool 的连接方法
            with self._get_milvus_client(self.runtime.credentials) as milvus_http_client:
                logger.info("✅ [DEBUG] Successfully connected to Milvus for BM25 search")
                
                # 为 BM25 搜索创建专用的 MilvusClient
                credentials = self.runtime.credentials
                uri = credentials.get("uri")
                token = credentials.get("token")
                database = credentials.get("database", "default")
                
                # 确保 URI 格式正确
                if not uri.startswith(("http://", "https://")):
                    uri = f"http://{uri}"
                uri = uri.rstrip('/')
                
                # 创建用于 BM25 的 MilvusClient (需要SDK客户端)
                logger.info(f"Creating MilvusClient for BM25: {uri}")
                client = MilvusClient(
                    uri=uri,
                    token=token if token else None,
                    db_name=database
                )
                
                try:
                    # 检查集合是否存在
                    if not client.has_collection(collection_name):
                        raise ValueError(f"Collection '{collection_name}' does not exist")
                    
                    # 构建搜索参数
                    search_params = {
                        "metric_type": "BM25",
                        "params": {
                            "bm25_k1": bm25_k1,
                            "bm25_b": bm25_b
                        }
                    }
                    
                    # 处理输出字段
                    output_fields_list = None
                    if output_fields:
                        output_fields_list = [field.strip() for field in output_fields.split(',')]
                    
                    # 执行BM25搜索
                    logger.info(f"Performing BM25 search: '{query_text}' in collection '{collection_name}'")
                    
                    search_results = client.search(
                        collection_name=collection_name,
                        data=[query_text],  # BM25搜索直接使用文本
                        limit=limit,
                        search_params=search_params,
                        output_fields=output_fields_list,
                        filter=filter_expr if filter_expr else None
                    )
                    
                    # 处理搜索结果
                    results = []
                    if search_results and len(search_results) > 0:
                        for hit in search_results[0]:  # 第一个查询的结果
                            result_item = {
                                "id": hit.get("id"),
                                "score": hit.get("distance", hit.get("score", 0)),
                                "entity": hit.get("entity", {})
                            }
                            
                            # 将entity中的字段提升到顶层以便访问
                            if hit.get("entity"):
                                for key, value in hit["entity"].items():
                                    if key not in result_item:
                                        result_item[key] = value
                            
                            results.append(result_item)
                    
                    result_data = {
                        "operation": "bm25_search",
                        "query_text": query_text,
                        "collection_name": collection_name,
                        "search_type": "BM25",
                        "total_results": len(results),
                        "results": results,
                        "search_params": {
                            "limit": limit,
                            "text_field": text_field,
                            "bm25_k1": bm25_k1,
                            "bm25_b": bm25_b,
                            "filter": filter_expr,
                            "output_fields": output_fields
                        },
                        "message": f"Found {len(results)} results using BM25 algorithm in collection {collection_name}"
                    }
                    
                    logger.info(f"✅ [DEBUG] BM25 search completed: {len(results)} results found")
                    yield from self._create_success_message(result_data)
                    
                finally:
                    # 关闭BM25客户端连接
                    try:
                        client.close()
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"❌ [DEBUG] Error in BM25 search: {type(e).__name__}: {str(e)}", exc_info=True)
            yield from self._handle_error(e)
    
    def _handle_error(self, error: Exception) -> Generator[ToolInvokeMessage]:
        """统一的错误处理"""
        logger.error(f"🚨 [DEBUG] _handle_error() called with: {type(error).__name__}: {str(error)}")
        error_msg = str(error)
        response = {
            "success": False,
            "error": error_msg,
            "error_type": type(error).__name__
        }
        logger.debug(f"📤 [DEBUG] Sending error response: {response}")
        yield self.create_json_message(response)
    
    def _create_success_message(self, data: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """创建成功响应消息"""
        logger.debug(f"🎉 [DEBUG] _create_success_message() called with data: {data}")
        response = {
            "success": True,
            **data
        }
        logger.debug(f"📤 [DEBUG] Sending success response: {response}")
        yield self.create_json_message(response)


# 在模块级别添加调试信息
logger.info("📦 [DEBUG] milvus_bm25_search.py module loaded")