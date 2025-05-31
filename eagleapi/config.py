"""
Configuration settings for the Eagle Framework.

This module handles loading and validating configuration from environment variables.
"""
from pydantic import Field, PostgresDsn, validator
from pydantic_settings import BaseSettings  
from typing import Optional, Dict, Any, Union
from functools import lru_cache
import secrets
import os

class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    APP_NAME: str = "Eagle API"
    DEBUG: bool = False
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key for cryptographic operations"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES:int =180
    
    # Database settings
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./eagle.db",
        env="DATABASE_URL",
        description="Database connection URL. Defaults to SQLite in the current directory."
    )
    
    # Validate database URL
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        
        # If no URL is provided, use SQLite
        return "sqlite+aiosqlite:///./eagle.db"
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Eagle API"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = ["*"]
    
    # Security
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    class Config:
        case_sensitive = True
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get application settings with caching."""
    return Settings()


# Global settings instance
settings = get_settings()

# Export settings
__all__ = ["settings", "get_settings"]
