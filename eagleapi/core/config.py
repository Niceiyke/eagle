# core/config.py
"""
Configuration settings for Eagle Framework.
"""
import os
from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings   

class Settings(BaseSettings):
    """
    Centralized application settings for Eagle Framework.
    All settings can be overridden by environment variables (see .env.example).
    """
    # --- Application ---
    APP_NAME: str = "Eagle Framework"
    DEBUG: bool = False
    PROJECT_NAME: str = "Eagle API"
    API_V1_STR: str = "/api/v1"

    # --- Database ---
    DATABASE_URL: str = "sqlite+aiosqlite:///./eagle.db"  # EAGLE_DATABASE_URL
    ECHO_SQL: bool = False  # EAGLE_ECHO_SQL

    # --- Security & Auth ---
    SECRET_KEY: str = "your-secret-key-change-this-in-production!"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30  # JWT access token expiry (minutes)
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30  # JWT refresh token expiry (days)
    EMAIL_VERIFICATION_EXPIRE_HOURS: int = 24
    PASSWORD_RESET_EXPIRE_HOURS: int = 1

    # --- Superuser (initial setup) ---
    SUPERUSER_EMAIL: str = "superuser@example.com"
    SUPERUSER_PASSWORD: str = "superuser-password"

    # --- Server ---
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DOCS_ENABLED: bool = True
    RELOAD: bool = False
    WORKERS: int = 1

    ENV: str = "development"

    # --- CORS ---
    CORS_ORIGINS: List[str] = ["*"]

    # --- Admin ---
    ADMIN_ENABLED: bool = True  # EAGLE_ADMIN_ENABLED
    ADMIN_PATH: str = "/admin"  # EAGLE_ADMIN_PATH

    # --- Cache ---
    CACHE_BACKEND: str = "memory"  # Options: memory, redis
    REDIS_URL: Optional[str] = None  # REDIS_URL or CACHE_REDIS_URL
    CACHE_DEFAULT_TTL: int = 3600

    # --- Timezone ---
    USE_TIMEZONE: bool = False
    TIMEZONE: str = "UTC"

    # --- OAuth (social login) ---
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GOOGLE_REDIRECT_URI: Optional[str] = None
    GITHUB_CLIENT_ID: Optional[str] = None
    GITHUB_CLIENT_SECRET: Optional[str] = None

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