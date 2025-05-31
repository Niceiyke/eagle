# Eagle Database Module

This module provides a simple, powerful interface for database operations in the Eagle Framework.

## Features

- **Async SQLAlchemy 2.0+ Support**: Built on top of SQLAlchemy's async features
- **Automatic Session Management**: Handles sessions and transactions automatically
- **Migrations**: Built-in support for database migrations using Alembic
- **Type Hints**: Full Python type hint support for better IDE assistance
- **Multiple Database Support**: Works with SQLite, PostgreSQL, and MySQL

## Quick Start

### Installation

```bash
# Install with SQLite support (included by default)
pip install eagleapi

# For PostgreSQL support
pip install eagleapi[postgresql]

# For MySQL support
pip install eagleapi[mysql]
```

### Basic Usage

```python
from eagleapi.db import Base, BaseModel, db, get_db
from sqlalchemy import Column, String, Integer

# Define your models
class User(BaseModel):
    __tablename__ = "users"
    
    email: str = Column(String, unique=True, index=True)
    hashed_password: str = Column(String)
    full_name: str = Column(String)
    is_active: bool = Column(Boolean, default=True)

# Use the database
async def create_user(email: str, password: str, full_name: str):
    async with db.get_session() as session:
        user = User(email=email, hashed_password=hash_password(password), full_name=full_name)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user
```

## Configuration

Configure your database using environment variables:

```bash
# SQLite (default)
EAGLE_DATABASE_URL=sqlite+aiosqlite:///./eagle.db

# PostgreSQL
EAGLE_DATABASE_URL=postgresql+asyncpg://user:password@localhost/dbname

# MySQL
EAGLE_DATABASE_URL=mysql+aiomysql://user:password@localhost/dbname

# Enable SQL query logging
EAGLE_ECHO_SQL=true
```

## Database Migrations

Eagle provides a simple CLI for managing database migrations:

```bash
# Initialize migrations (first time only)
eagle db init

# Create a new migration
eagle db migrate -m "Add users table"

# Apply migrations
eagle db upgrade

# Rollback last migration
eagle db downgrade -1

# Show migration history
eagle db history

# Show current migration
eagle db current
```

## Advanced Usage

### Using the Database Session

You can get a database session in two ways:

1. **Using the context manager (recommended):**
   ```python
   async with db.get_session() as session:
       # Use the session
       result = await session.execute(select(User))
       users = result.scalars().all()
   ```

2. **Using the FastAPI dependency (in route handlers):**
   ```python
   from fastapi import Depends
   from sqlalchemy.ext.asyncio import AsyncSession
   
   @app.get("/users")
   async def get_users(session: AsyncSession = Depends(get_db)):
       result = await session.execute(select(User))
       return result.scalars().all()
   ```

### Transaction Management

Transactions are handled automatically when using the context manager or FastAPI dependency. For manual control:

```python
async with db.get_session() as session:
    try:
        # Start a transaction
        async with session.begin():
            # Your database operations here
            session.add(some_object)
            await session.flush()
            
            # This will be committed if no exceptions are raised
    except Exception:
        # Transaction is automatically rolled back on exception
        raise
```

## Best Practices

1. **Always use async/await** with database operations
2. **Keep sessions short-lived** - create a new session for each operation or request
3. **Use transactions** for operations that modify multiple records
4. **Handle exceptions** to ensure proper cleanup of resources
5. **Use type hints** for better IDE support and code clarity

## Troubleshooting

### Common Issues

1. **"Database not initialized" error**
   - Make sure you've called `db.setup()` at application startup
   - Check that your database URL is correctly configured

2. **Connection pool exhausted**
   - Increase the pool size in your database configuration
   - Make sure to properly close database sessions

3. **Migration issues**
   - Make sure your models match your database schema
   - Check the migration scripts for any errors

For more help, check the [SQLAlchemy documentation](https://docs.sqlalchemy.org/).
