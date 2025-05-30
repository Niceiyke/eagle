I want to create a new Python web framework called "Eagle" that extends FastAPI with enterprise-grade features. The framework should be built with the following specifications:

## Core Requirements
1. Built on top of FastAPI for high performance and async support
2. Use uv for dependency management instead of poetry
3. Follow modern Python packaging standards with pyproject.toml
4. Support Python 3.8+ with full type hints

## Key Features to Implement
1. **Core Framework**
   - Extension system for modular functionality
   - Built-in dependency injection
   - Configuration management
   - Middleware support

2. **Database Layer**
   - SQLAlchemy 2.0+ integration
   - Async database support
   - Migration system (Alembic)
   - Model system with Pydantic integration

3. **Admin Interface**
   - Automatic CRUD interface
   - Customizable dashboard
   - Model registration system
   - Form customization

4. **CLI Tool**
   - Project scaffolding
   - Database migrations
   - Development server
   - Code generation

5. **Authentication & Authorization**
   - JWT support
   - OAuth2 integration
   - Role-based access control
   - Permission system

## Project Structure
eagle/
├── __init__.py
├── core/           # Core framework code
├── extensions/     # Built-in extensions
├── db/             # Database related code
├── auth/           # Authentication system
├── admin/          # Admin interface
├── cli/            # Command line interface
└── tests/          # Test suite

## Technical Specifications
1. Use uv for dependency management
2. Include comprehensive type hints
3. Follow PEP 8 and modern Python practices
4. Include unit tests with pytest
5. Document all public APIs
6. Include example projects

## Deliverables
1. Complete project structure
2. Core framework implementation
3. Basic extension system
4. Database integration
5. Admin interface scaffolding
6. CLI tool with basic commands
7. Documentation and examples

## Additional Context
- Focus on developer experience
- Prioritize performance and scalability
- Include proper error handling
- Follow security best practices
- Make it extensible and modular

Please generate the initial implementation including all necessary files and configurations.