"""
Token-related Pydantic models for authentication.
"""
from pydantic import BaseModel

class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Token data model for JWT payload."""
    email: str | None = None
