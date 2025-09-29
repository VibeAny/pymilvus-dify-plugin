"""
PyMilvus-based Provider for Milvus Vector Database

This provider integrates with Dify's model system for text embedding functionality.
Only Milvus connection credentials are required - embedding models are managed by Dify.
"""
from typing import Any

# Import logging and custom handler for debugging
import logging
from dify_plugin.config.logger_format import plugin_logger_handler

# Set up logging with the custom handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(plugin_logger_handler)

# Configure proxy BEFORE importing PyMilvus to ensure proper initialization
import os
http_proxy = os.environ.get('SANDBOX_HTTP_PROXY') or os.environ.get('SSRF_PROXY_HTTP_URL')
https_proxy = os.environ.get('SANDBOX_HTTPS_PROXY') or os.environ.get('SSRF_PROXY_HTTPS_URL')

if http_proxy or https_proxy:
    logger.info(f"🌐 [DEBUG] Setting up proxy BEFORE PyMilvus import - HTTP: {http_proxy}, HTTPS: {https_proxy}")
    if http_proxy:
        os.environ['HTTP_PROXY'] = http_proxy
    if https_proxy:
        os.environ['HTTPS_PROXY'] = https_proxy
    logger.info("✅ [DEBUG] Global proxy environment configured")
else:
    logger.info("ℹ️ [DEBUG] No proxy environment detected")

def test_milvus_connection(uri: str, user: str, password: str, database: str) -> tuple:
    """Test MilvusClient connection - standalone function for multiprocessing"""
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(
            uri=uri,
            user=user, 
            password=password,
            db_name=database
        )
        # Test the connection
        collections = client.list_collections()
        return ('success', len(collections))
    except Exception as e:
        return ('error', str(e))

# Use conditional imports for testing compatibility
logger.info("🔄 [DEBUG] About to import PyMilvus and Dify plugin modules")
try:
    logger.info("🔄 [DEBUG] Importing PyMilvus MilvusClient...")
    from pymilvus import MilvusClient
    logger.info("✅ [DEBUG] PyMilvus MilvusClient imported successfully")
    
    logger.info("🔄 [DEBUG] Importing Dify ToolProvider...")
    from dify_plugin import ToolProvider
    logger.info("✅ [DEBUG] Dify ToolProvider imported successfully")
    
    logger.info("🔄 [DEBUG] Importing Dify ToolProviderCredentialValidationError...")
    from dify_plugin.errors.tool import ToolProviderCredentialValidationError
    logger.info("✅ [DEBUG] Dify ToolProviderCredentialValidationError imported successfully")
    
    logger.info("🔧 [DEBUG] Successfully imported ALL PyMilvus and Dify plugin modules")
except ImportError as e:
    # For testing - these will be mocked
    logger.warning(f"⚠️ [DEBUG] Import failed, using mock objects: {e}")
    MilvusClient = None
    ToolProvider = object
    ToolProviderCredentialValidationError = Exception


class MilvusProvider(ToolProvider):
    def __init__(self):
        super().__init__()
        logger.info("🚀 [DEBUG] MilvusProvider initialized")
    
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        """Validate credentials using PyMilvus client"""
        logger.info("🔍 [DEBUG] Starting credential validation")
        logger.info(f"📋 [DEBUG] Received credentials keys: {list(credentials.keys())}")
        
        try:
            # Only validate Milvus connection - embedding models are managed by Dify
            logger.info("🏁 [DEBUG] About to start Milvus connection validation")
            self._validate_milvus_connection(credentials)
            logger.info("✅ [DEBUG] Credential validation completed successfully")
        except Exception as e:
            logger.error(f"❌ [DEBUG] Credential validation failed: {type(e).__name__}: {str(e)}")
            raise
    
    def _validate_milvus_connection(self, credentials: dict[str, Any]) -> None:
        """Validate Milvus connection using PyMilvus with timeout protection"""
        logger.info("🔌 [DEBUG] Starting Milvus connection validation")
        
        # Check if MilvusClient was imported successfully
        if MilvusClient is None:
            logger.error("❌ [DEBUG] MilvusClient is None - PyMilvus not available")
            raise ToolProviderCredentialValidationError("PyMilvus library is not available")
        
        uri = credentials.get("uri")
        user = credentials.get("user") 
        password = credentials.get("password")
        database = credentials.get("database", "default")
        
        logger.info(f"🌐 [DEBUG] Connection details - URI: {uri}, User: {user}, Database: {database}")
        
        # Check required fields
        if not uri:
            logger.error("❌ [DEBUG] Missing URI")
            raise ToolProviderCredentialValidationError("Milvus URI is required")
        if not user:
            logger.error("❌ [DEBUG] Missing username")
            raise ToolProviderCredentialValidationError("Username is required")
        if not password:
            logger.error("❌ [DEBUG] Missing password")
            raise ToolProviderCredentialValidationError("Password is required")
        
        # Validate and fix URI format for PyMilvus gRPC client
        logger.info("🔧 [DEBUG] Validating URI format for PyMilvus gRPC client")
        
        # PyMilvus requires URI to start with [unix, http, https, tcp] or be a local .db file
        original_uri = uri
        if uri.startswith("https://"):
            # Keep HTTPS format - PyMilvus supports it
            logger.info(f"✅ [DEBUG] URI already in HTTPS format: {uri}")
        elif uri.startswith("http://"):
            # Keep HTTP format - PyMilvus supports it  
            logger.info(f"✅ [DEBUG] URI already in HTTP format: {uri}")
        elif uri.startswith("tcp://"):
            # Keep TCP format - PyMilvus supports it
            logger.info(f"✅ [DEBUG] URI already in TCP format: {uri}")
        elif uri.startswith("unix://"):
            # Keep Unix format - PyMilvus supports it
            logger.info(f"✅ [DEBUG] URI already in Unix format: {uri}")
        else:
            # Plain host:port format - convert to HTTPS (most common for cloud Milvus)
            if ":" in uri:
                logger.info(f"🔧 [DEBUG] Converting plain host:port to HTTPS: {uri}")
                uri = f"https://{uri}"
            else:
                # Plain hostname - add default port and HTTPS
                logger.info(f"🔧 [DEBUG] Converting plain hostname to HTTPS with port 443: {uri}")
                uri = f"https://{uri}:443"
                
        logger.info(f"✅ [DEBUG] Final URI for PyMilvus: {uri} (original: {original_uri})")
        
        logger.info("📋 [DEBUG] All required fields present, proceeding with connection test")
        
        # Direct connection test with global proxy configuration
        logger.info("🔧 [DEBUG] Starting direct connection test")
        try:
            logger.info("🧪 [DEBUG] Testing PyMilvus import in current context...")
            import pymilvus
            logger.info(f"✅ [DEBUG] PyMilvus version in context: {pymilvus.__version__}")
            
            # Test MilvusClient class access
            logger.info("🧪 [DEBUG] Testing MilvusClient class access...")
            client_class = pymilvus.MilvusClient
            logger.info(f"✅ [DEBUG] MilvusClient class: {client_class}")
            
            logger.info("🔧 [DEBUG] About to create PyMilvus client with parameters...")
            logger.info(f"🔧 [DEBUG] URI: {uri}")
            logger.info(f"🔧 [DEBUG] User: {user}")
            logger.info(f"🔧 [DEBUG] Database: {database}")
            logger.info("🔧 [DEBUG] Password: [MASKED]")
            
            # First test basic network connectivity before attempting MilvusClient
            logger.info("🌐 [DEBUG] Testing basic network connectivity to domain...")
            
            import socket
            import urllib.parse
            
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
                raise ToolProviderCredentialValidationError(f"Cannot resolve hostname {hostname}: {dns_e}")
            
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
                raise ToolProviderCredentialValidationError(f"Cannot connect to {hostname}:{port}: {tcp_e}")
            
            # Reset socket timeout for MilvusClient
            socket.setdefaulttimeout(None)
            
            # Create PyMilvus client with signal-based timeout (more reliable than multiprocessing)
            logger.info("🔧 [DEBUG] Network tests passed, attempting MilvusClient with signal-based timeout...")
            
            import signal
            
            def timeout_handler(signum, frame):
                logger.error("⏰ [DEBUG] Signal timeout triggered")
                raise TimeoutError("MilvusClient creation timed out")
            
            try:
                # Set signal-based timeout
                logger.info("⏰ [DEBUG] Setting 8-second signal timeout...")
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(8)  # 8 second timeout
                
                # Try direct client creation with timeout protection
                logger.info("🔄 [DEBUG] Creating MilvusClient directly...")
                result_type, result_value = test_milvus_connection(uri, user, password, database)
                
                # Cancel the alarm
                signal.alarm(0)
                
                if result_type == 'success':
                    logger.info(f"✅ [DEBUG] MilvusClient connection successful, found {result_value} collections")
                else:
                    logger.error(f"❌ [DEBUG] MilvusClient creation failed: {result_value}")
                    raise RuntimeError(result_value)
                    
            except TimeoutError:
                signal.alarm(0)  # Cancel alarm
                logger.error("⏰ [DEBUG] MilvusClient creation timed out after 8 seconds")
                raise TimeoutError("MilvusClient creation timed out after 8 seconds")
            except Exception as e:
                signal.alarm(0)  # Cancel alarm
                logger.error(f"❌ [DEBUG] MilvusClient creation failed: {e}")
                raise e
            logger.info("🎉 [DEBUG] Connection validation successful!")
            
        except Exception as e:
            logger.error(f"❌ [DEBUG] Connection test failed: {type(e).__name__}: {str(e)}")
            logger.error(f"❌ [DEBUG] Full exception details: {repr(e)}")
            # Provide user-friendly error messages based on error type
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                logger.error("🚫 [DEBUG] Raising authentication error")
                raise ToolProviderCredentialValidationError("Authentication failed. Please check username and password.")
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                logger.error("🌐 [DEBUG] Raising connection error")
                raise ToolProviderCredentialValidationError("Cannot connect to Milvus server. Please check URI and network connectivity.")
            else:
                logger.error("⚠️ [DEBUG] Raising generic Milvus error")
                raise ToolProviderCredentialValidationError(f"Milvus connection failed: {error_msg}")
        
        logger.info("🏁 [DEBUG] _validate_milvus_connection completed successfully")