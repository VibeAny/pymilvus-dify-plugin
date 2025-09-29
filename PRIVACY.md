# Privacy Policy for Milvus Vector Database Plugin

**Last Updated:** January 27, 2025

## Overview

The Milvus Vector Database Plugin ("Plugin") is designed to integrate Milvus vector database services with the Dify platform. This privacy policy explains what information is collected, how it is used, and how it is protected.

## Information Collection

### User-Provided Data

The Plugin processes the following types of data that you explicitly provide:

- **Connection Credentials**: Milvus server URI, username, password, and database name
- **Embedding API Keys**: OpenAI API keys or Azure OpenAI credentials for text embedding services
- **Vector Data**: Numerical vectors and associated metadata that you store in your Milvus collections
- **Text Content**: Text documents and queries that are processed for embedding or search operations

### Automatically Collected Data

The Plugin does not automatically collect any personal information, analytics, or usage statistics.

## Data Usage

### Primary Functions

Your data is used exclusively for the following purposes:

- **Database Operations**: Connecting to and managing your Milvus vector database
- **Text Processing**: Converting text to vectors using your configured embedding providers
- **Search Operations**: Performing vector similarity searches and BM25 keyword searches
- **Collection Management**: Creating, updating, and managing your vector collections

### Data Processing

- **Local Processing**: All plugin operations are performed within your Dify environment
- **Third-Party APIs**: Text data may be sent to OpenAI or Azure OpenAI for embedding generation when you use text embedding features
- **No Data Storage**: The plugin does not store any of your data; all information is processed in real-time

## Data Sharing

### With Third Parties

- **Milvus Server**: Your data is transmitted to your configured Milvus server for storage and retrieval
- **Embedding Providers**: When using text embedding features, text content is sent to your configured provider (OpenAI or Azure OpenAI) for vector generation
- **No Other Sharing**: We do not share your data with any other third parties

### Service Provider Privacy Policies

When using external services, their privacy policies apply:
- **OpenAI**: [https://openai.com/privacy/](https://openai.com/privacy/)
- **Azure OpenAI**: [https://privacy.microsoft.com/privacystatement](https://privacy.microsoft.com/privacystatement)
- **Milvus**: Data is processed according to your own Milvus server configuration

## Data Security

### Credential Protection

- **Secure Storage**: All credentials are stored securely within the Dify platform
- **Encrypted Transmission**: All communications use secure protocols (HTTPS/TLS)
- **No Hardcoding**: API keys and credentials are never hardcoded in the plugin

### Data Handling

- **Minimal Processing**: Only necessary data is processed for requested operations
- **No Persistence**: The plugin does not maintain permanent storage of your data
- **Error Handling**: Error messages are designed to be informative without exposing sensitive data

## User Rights and Control

### Data Control

- **Full Ownership**: You maintain complete ownership and control of your data
- **Configuration Control**: You can configure which embedding providers to use
- **Connection Control**: You can modify or remove connection credentials at any time

### Data Deletion

- **Plugin Removal**: Uninstalling the plugin removes all stored credentials from Dify
- **Data Persistence**: Your data remains in your Milvus database and is not affected by plugin removal

## Compliance and Regional Considerations

### GDPR Compliance (EU Users)

If you are in the European Union, you have additional rights under GDPR:
- **Right to Access**: Request information about data processing
- **Right to Rectification**: Correct inaccurate data
- **Right to Erasure**: Request deletion of your data
- **Right to Portability**: Export your data in a structured format

### CCPA Compliance (California Users)

California residents have rights under the CCPA:
- **Right to Know**: Information about data collection and use
- **Right to Delete**: Request deletion of personal information
- **Right to Opt-Out**: Opt-out of the sale of personal information (not applicable as we don't sell data)

## Updates to This Policy

We may update this privacy policy from time to time. When we do:
- **Notification**: Changes will be reflected in the "Last Updated" date
- **Material Changes**: Significant changes will be communicated through appropriate channels
- **Continued Use**: Your continued use of the plugin after updates constitutes acceptance of the new policy

## Contact Information

For questions about this privacy policy or data handling practices:

- **Support**: Contact through the plugin's designated support channel in the Dify marketplace
- **Data Concerns**: Address privacy-specific questions to the plugin maintainer

## Children's Privacy

This plugin is not intended for use by children under 13 years of age. We do not knowingly collect personal information from children under 13.

## International Data Transfers

Data may be transferred internationally when:
- Using cloud-based Milvus services
- Using OpenAI or Azure OpenAI embedding services
- All transfers comply with applicable data protection laws

---

**Note**: This privacy policy applies specifically to the Milvus Vector Database Plugin. For information about Dify platform privacy practices, please refer to Dify's privacy policy.