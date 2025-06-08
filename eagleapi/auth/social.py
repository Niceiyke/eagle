
# auth/social.py
"""
Social authentication providers for Eagle Framework.
"""
import httpx
import secrets
from typing import Dict, Any, Optional
from fastapi import HTTPException, status, Depends, APIRouter
from sqlalchemy.ext.asyncio import AsyncSession
from ..core.config import settings
from ..db import get_db
from . import AuthUser, create_user, UserCreate, AuthProvider

class SocialAuthProvider:
    """Base social authentication provider."""
    
    def __init__(self):
        self.client_id = None
        self.client_secret = None
        self.redirect_uri = None
    
    async def get_user_info(self, token: str) -> Dict[str, Any]:
        """Get user info from social provider."""
        raise NotImplementedError
    
    async def exchange_code_for_token(self, code: str) -> str:
        """Exchange authorization code for access token."""
        raise NotImplementedError

class GoogleAuthProvider(SocialAuthProvider):
    """Google OAuth2 authentication provider."""
    
    def __init__(self):
        super().__init__()
        self.client_id = getattr(settings, 'GOOGLE_CLIENT_ID', None)
        self.client_secret = getattr(settings, 'GOOGLE_CLIENT_SECRET', None)
        self.redirect_uri = getattr(settings, 'GOOGLE_REDIRECT_URI', None)
    
    async def get_user_info(self, token: str) -> Dict[str, Any]:
        """Get user info from Google."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get user info from Google"
                )
            return response.json()
    
    async def exchange_code_for_token(self, code: str) -> str:
        """Exchange authorization code for access token."""
        async with httpx.AsyncClient() as client:
            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self.redirect_uri,
            }
            response = await client.post("https://oauth2.googleapis.com/token", data=data)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange code for token"
                )
            return response.json()["access_token"]

class GitHubAuthProvider(SocialAuthProvider):
    """GitHub OAuth2 authentication provider."""
    
    def __init__(self):
        super().__init__()
        self.client_id = getattr(settings, 'GITHUB_CLIENT_ID', None)
        self.client_secret = getattr(settings, 'GITHUB_CLIENT_SECRET', None)
    
    async def get_user_info(self, token: str) -> Dict[str, Any]:
        """Get user info from GitHub."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"token {token}"}
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get user info from GitHub"
                )
            return response.json()
    
    async def exchange_code_for_token(self, code: str) -> str:
        """Exchange authorization code for access token."""
        async with httpx.AsyncClient() as client:
            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
            }
            headers = {"Accept": "application/json"}
            response = await client.post(
                "https://github.com/login/oauth/access_token",
                data=data,
                headers=headers
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange code for token"
                )
            return response.json()["access_token"]

# Social auth service
social_providers = {
    AuthProvider.GOOGLE: GoogleAuthProvider(),
    AuthProvider.GITHUB: GitHubAuthProvider(),
}

async def authenticate_social_user(
    provider: AuthProvider,
    code: str,
    db: AsyncSession
) -> AuthUser:
    """Authenticate user via social provider."""
    if provider not in social_providers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider {provider} not supported"
        )
    
    provider_instance = social_providers[provider]
    token = await provider_instance.exchange_code_for_token(code)
    user_info = await provider_instance.get_user_info(token)
    
    # Extract user data based on provider
    if provider == AuthProvider.GOOGLE:
        email = user_info["email"]
        full_name = user_info.get("name", "")
        social_id = user_info["id"]
        username = email.split("@")[0] + "_google"
    elif provider == AuthProvider.GITHUB:
        email = user_info.get("email", "")
        full_name = user_info.get("name", "")
        social_id = str(user_info["id"])
        username = user_info["login"] + "_github"
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email not provided by social provider"
        )
    
    # Check if user exists
    from . import get_user_by_email
    user = await get_user_by_email(db, email)
    
    if not user:
        # Create new user
        user_create = UserCreate(
            username=username,
            email=email,
            full_name=full_name,
            password=secrets.token_urlsafe(32),  # Random password
            confirm_password=secrets.token_urlsafe(32),
            is_verified=True  # Social accounts are pre-verified
        )
        user = await create_user(db, user_create)
        user.auth_provider = provider
        user.social_id = social_id
        await db.commit()
    else:
        # Update existing user's social info if needed
        if not user.social_id:
            user.social_id = social_id
            user.auth_provider = provider
            await db.commit()
    
    return user
