"""
Pure PyMilvus client wrapper for Milvus operations

This wrapper provides a clean interface to PyMilvus functionality,
specifically designed for the dify plugin with proper error handling
and BM25 support.
"""
from typing import Dict, List, Any, Optional
import sys

# Use conditional imports for testing compatibility
try:
    from pymilvus import MilvusClient, Function, FieldSchema, CollectionSchema, DataType, FunctionType
    from pymilvus.model.sparse import BM25EmbeddingFunction
except ImportError:
    # For testing - these will be mocked
    MilvusClient = type('MilvusClient', (), {})
    Function = type('Function', (), {})
    FieldSchema = type('FieldSchema', (), {})
    CollectionSchema = type('CollectionSchema', (), {})
    DataType = type('DataType', (), {})
    FunctionType = type('FunctionType', (), {})
    BM25EmbeddingFunction = type('BM25EmbeddingFunction', (), {})


class MilvusClientWrapper:
    """Wrapper around PyMilvus client with enhanced functionality"""
    
    def __init__(self, credentials: Dict[str, Any]):
        """Initialize wrapper with credentials"""
        self.credentials = credentials
        self._client = None  # Lazy initialization
    
    @property 
    def client(self) -> Any:
        """Get or create PyMilvus client (lazy initialization)"""
        if self._client is None:
            self._client = self._create_client(self.credentials)
        return self._client
    
    def _create_client(self, credentials: Dict[str, Any]) -> Any:
        """Create PyMilvus client from credentials with timeout protection"""
        import signal
        import socket
        import urllib.parse
        import logging
        
        logger = logging.getLogger(__name__)
        
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
        
        # Same network pre-checks as provider validation
        logger.info("🌐 [DEBUG] Testing basic network connectivity to domain...")
        
        # Parse the URI to get hostname and port
        parsed_uri = urllib.parse.urlparse(uri)
        hostname = parsed_uri.hostname
        port = parsed_uri.port or (443 if parsed_uri.scheme == 'https' else 80)
        
        logger.info(f"🔍 [DEBUG] Extracted hostname: {hostname}, port: {port}")
        
        # Test DNS resolution with timeout
        try:
            logger.info("🔎 [DEBUG] Testing DNS resolution...")
            socket.setdefaulttimeout(3.0)  # 3 second DNS timeout
            ip = socket.gethostbyname(hostname)
            logger.info(f"✅ [DEBUG] DNS resolution successful: {hostname} -> {ip}")
        except Exception as dns_e:
            logger.error(f"❌ [DEBUG] DNS resolution failed: {dns_e}")
            raise RuntimeError(f"Cannot resolve hostname {hostname}: {dns_e}")
        
        # Test TCP connection with timeout
        try:
            logger.info(f"🔌 [DEBUG] Testing TCP connection to {ip}:{port}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)  # 3 second connection timeout
            result = sock.connect_ex((ip, port))
            sock.close()
            if result == 0:
                logger.info(f"✅ [DEBUG] TCP connection successful to {ip}:{port}")
            else:
                logger.error(f"❌ [DEBUG] TCP connection failed to {ip}:{port}, error code: {result}")
                raise ConnectionError(f"Cannot connect to {ip}:{port}")
        except Exception as tcp_e:
            logger.error(f"❌ [DEBUG] TCP connection test failed: {tcp_e}")
            raise RuntimeError(f"Cannot connect to {hostname}:{port}: {tcp_e}")
        
        # Reset socket timeout for MilvusClient
        socket.setdefaulttimeout(None)
        
        # Create MilvusClient with signal-based timeout
        logger.info("🔧 [DEBUG] Network tests passed, attempting MilvusClient with timeout...")
        
        def timeout_handler(signum, frame):
            logger.error("⏰ [DEBUG] Signal timeout triggered in MilvusClientWrapper")
            raise TimeoutError("MilvusClient creation timed out in wrapper")
        
        try:
            # Set signal-based timeout
            logger.info("⏰ [DEBUG] Setting 8-second signal timeout...")
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(8)  # 8 second timeout
            
            # Create client
            client = MilvusClient(
                uri=uri,
                user=user,
                password=password,
                db_name=database
            )
            
            # Cancel the alarm
            signal.alarm(0)
            logger.info("✅ [DEBUG] MilvusClient created successfully in wrapper")
            return client
            
        except TimeoutError:
            signal.alarm(0)  # Cancel alarm
            logger.error("⏰ [DEBUG] MilvusClient creation timed out after 8 seconds in wrapper")
            raise TimeoutError("MilvusClient creation timed out after 8 seconds")
        except Exception as e:
            signal.alarm(0)  # Cancel alarm
            logger.error(f"❌ [DEBUG] MilvusClient creation failed in wrapper: {e}")
            raise e
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        return self.client.list_collections()
    
    def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists"""
        return self.client.has_collection(collection_name)
    
    def create_collection_with_schema(self, schema_config: Dict[str, Any]) -> None:
        """Create collection with schema configuration"""
        collection_name = schema_config["collection_name"]
        enable_bm25 = schema_config.get("enable_bm25", False)
        vector_field = schema_config["vector_field"]
        text_field = schema_config.get("text_field", {})
        
        # Create schema
        schema = self.client.create_schema(auto_id=True)
        
        # Add ID field (primary key)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        
        # Add text field if provided
        if text_field:
            text_field_name = text_field["name"]
            max_length = text_field["max_length"]
            schema.add_field(text_field_name, DataType.VARCHAR, max_length=max_length, enable_analyzer=True)
        
        # Add vector field
        vector_field_name = vector_field["name"]
        vector_dim = vector_field["dim"]
        schema.add_field(vector_field_name, DataType.FLOAT_VECTOR, dim=vector_dim)
        
        # Add sparse vector field for BM25 if enabled
        if enable_bm25:
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            
            # Add BM25 function
            if text_field:
                bm25_function = Function(
                    name="text_bm25_emb",
                    function_type=FunctionType.BM25,
                    input_field_names=[text_field_name],
                    output_field_names=["sparse_vector"]
                )
                schema.add_function(bm25_function)
        
        # Create index parameters
        index_params = self.client.prepare_index_params()
        index_params.add_index(vector_field_name, index_type="AUTOINDEX", metric_type="COSINE")
        
        if enable_bm25:
            index_params.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        
        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
    
    def insert(self, collection_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Insert data into collection"""
        return self.client.insert(
            collection_name=collection_name,
            data=data
        )
    
    def vector_search(self, collection_name: str, vectors: List[List[float]], 
                     limit: int = 10, output_fields: Optional[List[str]] = None) -> List[List[Dict[str, Any]]]:
        """Perform vector similarity search"""
        return self.client.search(
            collection_name=collection_name,
            data=vectors,
            anns_field="vector",
            limit=limit,
            output_fields=output_fields or []
        )
    
    def search(self, collection_name: str, data: List[List[float]], 
               anns_field: str = "vector", limit: int = 10, 
               output_fields: Optional[List[str]] = None, 
               filter: Optional[str] = None,
               search_params: Optional[Dict[str, Any]] = None,
               **kwargs) -> List[List[Dict[str, Any]]]:
        """Generic search method (alias for vector_search with additional parameters)"""
        return self.client.search(
            collection_name=collection_name,
            data=data,
            anns_field=anns_field,
            limit=limit,
            output_fields=output_fields or []
        )
    
    def bm25_search(self, collection_name: str, query_text: str,
                   limit: int = 10, output_fields: Optional[List[str]] = None) -> List[List[Dict[str, Any]]]:
        """Perform BM25 keyword search"""
        return self.client.search(
            collection_name=collection_name,
            data=[query_text],
            anns_field="sparse_vector",
            limit=limit,
            output_fields=output_fields or []
        )
    
    def hybrid_search(self, collection_name: str, vector: List[float], text: str,
                     limit: int = 10, vector_weight: float = 0.7, text_weight: float = 0.3,
                     output_fields: Optional[List[str]] = None) -> List[List[Dict[str, Any]]]:
        """Perform hybrid search (vector + BM25)"""
        # For now, implement as regular search - can be enhanced later
        # In real implementation, this would use Milvus hybrid search capabilities
        return self.client.search(
            collection_name=collection_name,
            data=[vector],
            anns_field="vector",
            limit=limit,
            output_fields=output_fields or []
        )
    
    def query(self, collection_name: str, filter: str,
             output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Query data with filter"""
        return self.client.query(
            collection_name=collection_name,
            filter=filter,
            output_fields=output_fields or []
        )
    
    def delete(self, collection_name: str, filter: str) -> Dict[str, Any]:
        """Delete data with filter"""
        return self.client.delete(
            collection_name=collection_name,
            filter=filter
        )
    
    def drop_collection(self, collection_name: str) -> None:
        """Drop collection"""
        self.client.drop_collection(collection_name)
    
    def describe_collection(self, collection_name: str) -> Dict[str, Any]:
        """Describe collection schema with serializable output"""
        raw_description = self.client.describe_collection(collection_name)
        
        # Convert protobuf objects to serializable format
        def convert_to_serializable(obj):
            """Recursively convert protobuf objects to basic Python types"""
            if hasattr(obj, '__dict__'):
                # Handle objects with attributes
                result = {}
                for key, value in obj.__dict__.items():
                    if not key.startswith('_'):  # Skip private attributes
                        result[key] = convert_to_serializable(value)
                return result
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                # Handle iterables (lists, repeated containers)
                try:
                    return [convert_to_serializable(item) for item in obj]
                except Exception:
                    return str(obj)
            elif hasattr(obj, '__class__') and 'google' in str(obj.__class__):
                # Handle Google protobuf objects by converting to string
                return str(obj)
            else:
                # Return basic types as-is
                return obj
        
        return convert_to_serializable(raw_description)