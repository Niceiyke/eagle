# Eagle API Framework

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Eagle is a modern, high-performance Python web framework built on top of FastAPI, designed for building enterprise-grade applications with minimal boilerplate.

## Features

- 🚀 **FastAPI-Powered**: Built on top of FastAPI for high performance and async support
- 🗄️ **Database Ready**: SQLAlchemy integration with async support
- 🔐 **Authentication**: JWT-based authentication built-in
- 🧪 **Testing**: Comprehensive test suite with pytest
- 🛠️ **CLI**: Command-line tools for common tasks
- 📦 **Modern Packaging**: Uses pyproject.toml and modern Python packaging standards
- 🛠️ **Batteries Included**: Comes with built-in database integration, authentication, admin interface, and more
- 🧩 **Extensible**: Modular architecture with a powerful extension system
- 🔒 **Secure**: Built with security best practices in mind
- 📦 **Modern Tooling**: Uses uv for dependency management and modern Python packaging
- 🧪 **Tested**: Comprehensive test suite with pytest

## Installation

Eagle requires Python 3.8+. Install it using pip:

```bash
# Install the latest version from PyPI
pip install eagleapi

# Or install from source
pip install git+https://github.com/yourusername/eagle.git

# For development installation
pip install -e .[dev]
```

## Quick Start

Create a new Eagle project:

```bash
# Create a new project
eagle new myproject

# Navigate to the project directory
cd myproject

# Install dependencies
pip install -r requirements.txt

# Set up your environment
cp .env.example .env

# Run the development server
python -m uvicorn app.main:app --reload
```

Visit http://localhost:8000 to see your Eagle application in action!

## Project Structure

```
myproject/
├── app/
│   ├── __init__.py
│   ├── main.py          # Main application module
│   ├── api/             # API routes
│   ├── core/            # Core functionality
│   ├── models/          # Database models
│   ├── schemas/         # Pydantic models
│   ├── services/        # Business logic
│   ├── static/          # Static files
│   └── templates/       # HTML templates
├── tests/               # Test suite
├── .env                 # Environment variables
├── .gitignore
├── pyproject.toml
└── README.md
```

## Core Components

### Application

```python
from eagle import Eagle

app = Eagle()

@app.get("/")
async def read_root():
    return {"message": "Welcome to Eagle Framework!"}
```

### Database Models

```python
from sqlalchemy import Column, Integer, String
from eagle.db import BaseModel

class User(BaseModel):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
```

### Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from eagle.auth import get_current_active_user, User

router = APIRouter()

@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user
```

## Extensions

Eagle comes with several built-in extensions:

- **Database**: SQLAlchemy integration with async support
- **Auth**: JWT-based authentication
- **Admin**: Automatic admin interface
- **CLI**: Command-line tools for development

## Examples

The `examples/` directory contains example applications that demonstrate various features of the Eagle framework:

### Basic Example

A simple example showing the core features of Eagle:

```bash
cd examples/basic_app
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://localhost:8000 in your browser to see it in action.

For more details, see the [examples/basic_app/README.md](examples/basic_app/README.md) file.

## Documentation

For complete documentation, please visit [Eagle Documentation](https://eagle-framework.readthedocs.io/).

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) to get started.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) - The web framework for building APIs with Python
- [SQLAlchemy](https://www.sqlalchemy.org/) - The Python SQL Toolkit and ORM
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation and settings management
- [Typer](https://typer.tiangolo.com/) - CLI framework

---

Built with ❤️ by the Eagle Team
