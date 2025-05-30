"""
Test script to verify database setup and table creation.
"""
import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, configure_mappers
from sqlalchemy.sql import text

# Add the project root to the Python path
sys.path.insert(0, '.')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Import models to ensure they are registered with SQLAlchemy
from eagleapi.db.base import Base
from eagleapi.db.user_model import User

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///test.db"

async def init_models():
    """Initialize database models and create tables."""
    logger.info("Initializing database models...")
    
    # Create engine
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=True,
        future=True
    )
    
    # Create async session factory
    async_session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    
    # Import all models to ensure they are registered with SQLAlchemy
    from eagleapi.db.user_model import User  # noqa: F401
    
    # Explicitly configure mappers
    logger.info("Configuring SQLAlchemy mappers...")
    configure_mappers()
    
    # Log registered tables
    logger.info("\nRegistered tables in metadata:")
    for table_name, table in Base.metadata.tables.items():
        logger.info(f"- {table_name} ({table})")
    
    # Create tables
    logger.info("\nCreating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    return engine, async_session_factory

async def verify_tables(engine):
    """Verify that tables were created correctly."""
    logger.info("\nVerifying database tables...")
    
    async with engine.connect() as conn:
        # Check if tables exist
        result = await conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
        tables = [row[0] for row in result]
        logger.info(f"Found tables: {tables}")
        
        if not tables:
            logger.error("❌ No tables found in the database!")
            return False
            
        if 'users' not in tables:
            logger.error("❌ 'users' table not found in the database!")
            return False
        
        # Check users table schema
        logger.info("\nSchema for 'users' table:")
        result = await conn.execute(text("PRAGMA table_info(users)"))
        columns = [dict(row) for row in result.mappings()]
        
        expected_columns = {
            'id', 'username', 'email', 'full_name', 'hashed_password',
            'is_active', 'is_superuser', 'created_at', 'updated_at'
        }
        
        found_columns = {col['name'] for col in columns}
        missing_columns = expected_columns - found_columns
        
        for col in columns:
            logger.info(f"  - {col['name']}: {col['type']} {'PRIMARY KEY' if col['pk'] else ''}")
        
        if missing_columns:
            logger.error(f"❌ Missing columns in 'users' table: {missing_columns}")
            return False
            
        logger.info("\n✅ All expected columns found in 'users' table")
        return True

async def test_db_setup():
    """Test database setup and table creation."""
    logger.info("\n" + "="*80)
    logger.info("TESTING DATABASE SETUP")
    logger.info("="*80)
    
    success = False
    engine = None
    
    try:
        # Initialize models and create tables
        engine, _ = await init_models()
        
        # Verify tables were created correctly
        success = await verify_tables(engine)
        
        if success:
            logger.info("\n✅ Database setup test passed!")
        else:
            logger.error("\n❌ Database setup test failed!")
            
    except Exception as e:
        logger.error(f"❌ Error during database setup: {e}", exc_info=True)
        success = False
        
    finally:
        if engine:
            await engine.dispose()
            
    return success

if __name__ == "__main__":
    import os
    
    # Clean up any existing test database
    if os.path.exists("test.db"):
        os.remove("test.db")
    
    # Run the test
    result = asyncio.run(test_db_setup())
    
    # Exit with appropriate status code
    exit(0 if result else 1)
