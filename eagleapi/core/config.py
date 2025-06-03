# core/config.py
"""
Configuration settings for Eagle Framework.
"""
import os
from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings   

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    APP_NAME: str = "Eagle Framework"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./eagle.db"
    ECHO_SQL: bool = False
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Authentication
    SUPERUSER_EMAIL: str = "superuser@example.com"
    SUPERUSER_PASSWORD: str = "superuser-password"
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # Admin
    ADMIN_ENABLED: bool = True
    ADMIN_PATH: str = "/admin"
    
    # Cache
    CACHE_BACKEND: str = "memory"
    REDIS_URL: Optional[str] = None
    CACHE_DEFAULT_TTL: int = 3600
    
    @field_validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        if v == "your-secret-key-change-this-in-production":
            import warnings
            warnings.warn("Using default SECRET_KEY in production is insecure!")
        return v
    
    @field_validator("CORS_ORIGINS")
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()