"""
Unit tests for new PyMilvus-based provider validation

Following TDD approach:
1. Red: Write failing tests first
2. Green: Write minimal code to make tests pass  
3. Refactor: Improve code while keeping tests green

Updated to remove embedding provider validation as it's now handled by Dify's model system.
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path to enable absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock dify_plugin for testing
class MockToolProviderCredentialValidationError(Exception):
    pass

class MockToolProvider:
    pass

# Add mock modules to sys.modules before any imports
sys.modules['dify_plugin'] = MagicMock()
sys.modules['dify_plugin.errors'] = MagicMock()
sys.modules['dify_plugin.errors.tool'] = MagicMock()

# Set the specific classes we need
sys.modules['dify_plugin'].ToolProvider = MockToolProvider
sys.modules['dify_plugin.errors.tool'].ToolProviderCredentialValidationError = MockToolProviderCredentialValidationError

# Now we can import the provider module
from provider.milvus import MilvusProvider


class TestMilvusProviderValidation:
    """Test the new PyMilvus-based provider validation"""
    
    @patch('provider.milvus.MilvusClient')
    def test_validate_credentials_success(self, mock_milvus_client_class, mock_credentials):
        """Test successful credential validation with PyMilvus (Dify model integration)"""
        # Setup mock Milvus client instance
        mock_client_instance = MagicMock()
        mock_client_instance.list_collections.return_value = ["collection1", "collection2"]
        mock_milvus_client_class.return_value = mock_client_instance
        
        provider = MilvusProvider()
        
        # Should not raise exception with valid credentials
        # Only Milvus credentials are validated, embedding models are handled by Dify
        provider._validate_credentials(mock_credentials)
        
        # Verify PyMilvus client was called correctly
        mock_milvus_client_class.assert_called_once_with(
            uri=mock_credentials["uri"],
            user=mock_credentials["user"],
            password=mock_credentials["password"],
            db_name=mock_credentials.get("database", "default")
        )
        mock_client_instance.list_collections.assert_called_once()
    
    def test_validate_credentials_missing_uri(self):
        """Test validation fails with missing URI"""
        provider = MilvusProvider()
        credentials = {
            "user": "root",
            "password": "test_password",
            "database": "default"
        }
        
        with pytest.raises(MockToolProviderCredentialValidationError, match="URI is required"):
            provider._validate_credentials(credentials)
    
    def test_validate_credentials_missing_user(self):
        """Test validation fails with missing user"""
        provider = MilvusProvider()
        credentials = {
            "uri": "https://milvus-api.roomwits.com",
            "password": "test_password",
            "database": "default"
        }
        
        with pytest.raises(MockToolProviderCredentialValidationError, match="Username is required"):
            provider._validate_credentials(credentials)
    
    def test_validate_credentials_missing_password(self):
        """Test validation fails with missing password"""
        provider = MilvusProvider()
        credentials = {
            "uri": "https://milvus-api.roomwits.com", 
            "user": "root",
            "database": "default"
        }
        
        with pytest.raises(MockToolProviderCredentialValidationError, match="Password is required"):
            provider._validate_credentials(credentials)
    
    @patch('provider.milvus.MilvusClient')
    def test_milvus_connection_authentication_failure(self, mock_client_class):
        """Test Milvus connection authentication failure"""
        # Mock authentication failure
        mock_client_class.side_effect = Exception("Authentication failed")
        
        provider = MilvusProvider()
        credentials = {
            "uri": "https://milvus-api.roomwits.com",
            "user": "wrong_user",
            "password": "wrong_password",
            "database": "default"
        }
        
        with pytest.raises(MockToolProviderCredentialValidationError, match="Authentication failed"):
            provider._validate_credentials(credentials)
    
    @patch('provider.milvus.MilvusClient')
    def test_milvus_connection_timeout(self, mock_client_class):
        """Test Milvus connection timeout"""
        # Mock connection timeout
        mock_client_class.side_effect = Exception("Connection timeout")
        
        provider = MilvusProvider()
        credentials = {
            "uri": "https://unreachable-milvus.com",
            "user": "root",
            "password": "password",
            "database": "default"
        }
        
        with pytest.raises(MockToolProviderCredentialValidationError, match="Cannot connect to Milvus server"):
            provider._validate_credentials(credentials)
    
    @patch('provider.milvus.MilvusClient')
    def test_milvus_connection_success_with_list_collections(self, mock_client_class):
        """Test successful Milvus connection by calling list_collections"""
        # Mock successful connection
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.list_collections.return_value = ["collection1", "collection2"]
        
        provider = MilvusProvider()
        credentials = {
            "uri": "https://milvus-api.roomwits.com",
            "user": "root",
            "password": "correct_password",
            "database": "default"
        }
        
        # Should not raise exception
        provider._validate_credentials(credentials)
        
        # Verify client was created with correct parameters
        mock_client_class.assert_called_once_with(
            uri="https://milvus-api.roomwits.com",
            user="root", 
            password="correct_password",
            db_name="default"
        )
        
        # Verify connection test was performed
        mock_client.list_collections.assert_called_once()
    
    @patch('provider.milvus.MilvusClient')
    def test_database_parameter_default(self, mock_milvus_client):
        """Test database parameter defaults to 'default' when not provided"""
        # Mock successful connection
        mock_client_instance = MagicMock()
        mock_milvus_client.return_value = mock_client_instance
        mock_client_instance.list_collections.return_value = []
        
        provider = MilvusProvider()
        credentials = {
            "uri": "https://milvus-api.roomwits.com",
            "user": "root", 
            "password": "password"
            # database not provided - should default to "default"
        }
        
        provider._validate_credentials(credentials)
        
        # Should have been called with default database
        mock_milvus_client.assert_called_once_with(
            uri="https://milvus-api.roomwits.com",
            user="root",
            password="password",
            db_name="default"
        )
    
    @patch('provider.milvus.MilvusClient')
    def test_simplified_validation_process(self, mock_milvus_client):
        """Test that validation only checks Milvus connection (embedding validation removed)"""
        # Mock successful connection
        mock_client_instance = MagicMock()
        mock_milvus_client.return_value = mock_client_instance
        mock_client_instance.list_collections.return_value = []
        
        provider = MilvusProvider()
        credentials = {
            "uri": "https://milvus-api.roomwits.com",
            "user": "root",
            "password": "password",
            "database": "default"
        }
        
        # Should only validate Milvus connection, no embedding validation
        provider._validate_credentials(credentials)
        
        # Only Milvus client should have been called
        mock_milvus_client.assert_called_once()
        mock_client_instance.list_collections.assert_called_once()


class TestProviderConfigurationSchema:
    """Test the new provider configuration schema"""
    
    def test_provider_yaml_structure(self):
        """Test provider.yaml has correct structure for Dify model integration"""
        import yaml
        from pathlib import Path
        
        # Test the new provider configuration
        provider_path = Path("provider/milvus.yaml")
        
        if provider_path.exists():
            with open(provider_path) as f:
                config = yaml.safe_load(f)
            
            # Test new credential schema
            credentials = config.get("credentials_for_provider", {})
            
            # Required Milvus credentials only
            assert "uri" in credentials
            assert "user" in credentials  
            assert "password" in credentials
            assert "database" in credentials
            
            # Should not have embedding provider configuration (now handled by Dify)
            assert "embedding_provider" not in credentials
            assert "openai_api_key" not in credentials
            assert "azure_openai_endpoint" not in credentials
            assert "azure_openai_api_key" not in credentials
            assert "default_embedding_model" not in credentials
            
            # Should not have old HTTP token
            assert "token" not in credentials
            
            # Check URI field is properly configured
            uri_field = credentials["uri"]
            assert uri_field.get("required") is True
            assert "placeholder" in uri_field
            
            # Check updated identity
            identity = config.get("identity", {})
            assert identity.get("author") == "AI Plugin Developer"
            assert identity.get("name") == "milvus"
            
            # Check description mentions Dify integration
            description = identity.get("description", {}).get("en_US", "")
            assert "via Dify models" in description or "Dify model" in description
        else:
            pytest.skip("Provider YAML file not found - will be created during implementation")


# Fixtures for testing
@pytest.fixture
def mock_credentials():
    """Standard mock credentials for testing"""
    return {
        "uri": "https://milvus-api.roomwits.com",
        "user": "root",
        "password": "test_password",
        "database": "default"
    }