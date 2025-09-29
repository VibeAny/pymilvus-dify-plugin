# Contributing to PyMilvus Dify Plugin

Thank you for your interest in contributing to the PyMilvus Dify Plugin! We welcome contributions from the community and are grateful for your help in making this project better.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to creating a welcoming and inclusive environment. By participating, you are expected to uphold professional standards and treat all contributors with respect.

## Getting Started

### Prerequisites

- Python 3.8+
- Dify platform 1.6.0+
- Milvus 2.3.0+ server instance
- Git

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork the repo on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/pymilvus-dify-plugin.git
   cd pymilvus-dify-plugin
   ```

2. **Set up development environment**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   pip install -r test_requirements.txt
   ```

3. **Configure environment**
   ```bash
   # Copy environment template
   cp .env.example .env
   # Edit .env with your Milvus connection details
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/ -v
   ```

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Help us identify and fix issues
- **Feature enhancements**: Add new functionality or improve existing features
- **Documentation**: Improve documentation, examples, or tutorials
- **Testing**: Add test cases or improve test coverage
- **Performance**: Optimize code performance or resource usage

### Before You Start

1. **Check existing issues**: Look for existing issues or discussions about your idea
2. **Create an issue**: For new features or significant changes, create an issue first to discuss the approach
3. **Small fixes**: For small bug fixes or documentation improvements, you can submit a PR directly

## Pull Request Process

### 1. Create a Feature Branch

```bash
# Create and checkout a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Follow our [coding standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Test Your Changes

```bash
# Run unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run all tests with coverage
pytest --cov=. tests/
```

### 4. Commit Your Changes

```bash
# Add your changes
git add .

# Commit with a descriptive message
git commit -m "Add feature: description of your change"
```

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create a pull request on GitHub
```

### 6. Pull Request Requirements

Your PR should include:

- **Clear description**: Explain what changes you made and why
- **Issue reference**: Link to related issues (e.g., "Fixes #123")
- **Tests**: Add or update tests for your changes
- **Documentation**: Update relevant documentation
- **Changelog**: Add entry to CHANGELOG.md if applicable

## Issue Guidelines

### Reporting Bugs

When reporting bugs, please include:

- **Clear title**: Descriptive summary of the issue
- **Environment details**: Dify version, Python version, OS, etc.
- **Steps to reproduce**: Clear steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Error messages**: Include full error messages or stack traces
- **Configuration**: Relevant configuration details (sanitized)

### Feature Requests

For feature requests, please include:

- **Use case**: Describe the problem you're trying to solve
- **Proposed solution**: Your idea for addressing the use case
- **Alternatives considered**: Other approaches you've considered
- **Additional context**: Any other relevant information

## Coding Standards

### Python Code Style

- Follow [PEP 8](https://pep8.org/) Python style guidelines
- Use meaningful variable and function names
- Add docstrings for all public functions and classes
- Keep functions focused and reasonably sized

### Code Organization

- **tools/**: Plugin tool implementations
- **provider/**: Dify provider configuration
- **tests/**: All test files
- **docs/**: Documentation files
- **lib/**: Shared libraries and utilities

### Error Handling

- Use appropriate exception types
- Provide clear, actionable error messages
- Log errors appropriately
- Handle edge cases gracefully

### Documentation

- Use clear, concise language
- Include examples where helpful
- Update relevant documentation with your changes
- Follow existing documentation patterns

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests
└── conftest.py     # Shared test fixtures
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies appropriately
- Aim for high test coverage

### Test Categories

- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

## Documentation

### Types of Documentation

- **Code documentation**: Inline comments and docstrings
- **API documentation**: Tool parameter descriptions
- **User guides**: How-to guides and tutorials
- **Developer documentation**: Architecture and development guides

### Documentation Standards

- Use clear, concise language
- Include practical examples
- Keep documentation up-to-date with code changes
- Use proper markdown formatting

## Getting Help

If you need help or have questions:

- **GitHub Issues**: Create an issue for bugs or feature requests
- **GitHub Discussions**: For general questions and discussions
- **Documentation**: Check existing documentation first

## Recognition

Contributors will be recognized in:

- GitHub contributor list
- Release notes for significant contributions
- Documentation acknowledgments

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to PyMilvus Dify Plugin! Your efforts help make this project better for everyone.