import json
import logging
from typing import Any

from dify_plugin import Tool
from pymilvus import MilvusClient

logger = logging.getLogger(__name__)


class MilvusBM25SearchTool(Tool):
    def _invoke(self, user_id: str, tool_parameters: dict[str, Any]) -> str:
        """
        使用BM25算法执行关键词文本搜索
        """
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
            database = credentials.get("database", "default")
            
            if not uri:
                return json.dumps({
                    "success": False,
                    "error": "Milvus URI 未配置"
                }, ensure_ascii=False)
            
            # 创建Milvus客户端连接
            logger.info(f"Connecting to Milvus: {uri}")
            client = MilvusClient(
                uri=uri,
                token=token if token else None,
                db_name=database
            )
            
            # 检查集合是否存在
            if not client.has_collection(collection_name):
                return json.dumps({
                    "success": False,
                    "error": f"集合 {collection_name} 不存在"
                }, ensure_ascii=False)
            
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
            
            result = {
                "success": True,
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
                "message": f"在集合 {collection_name} 中使用BM25算法找到 {len(results)} 个相关结果"
            }
            
            logger.info(f"BM25 search completed: {len(results)} results found")
            return json.dumps(result, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"BM25 search failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": f"BM25搜索失败: {str(e)}"
            }, ensure_ascii=False)
        finally:
            # 关闭客户端连接
            try:
                if 'client' in locals():
                    client.close()
            except:
                pass