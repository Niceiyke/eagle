# auth/__init__.py
"""
Extended Authentication and authorization module for Eagle Framework.
Provides JWT authentication, OAuth2 integration, role-based access control,
refresh tokens, email verification, and social auth.
"""
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from enum import Enum

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, APIRouter, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import String, Boolean, DateTime, Text, select, and_
from sqlalchemy.orm import relationship

from ..db import BaseModel as DBBaseModel, get_db, Mapped, mapped_column,ForeignKey
from ..core.config import settings
from ..admin import register_model_to_admin


# === Enums ===
class AuthProvider(str, Enum):
    """Authentication providers."""
    LOCAL = "local"
    GOOGLE = "google"
    GITHUB = "github"
    FACEBOOK = "facebook"

class TokenType(str, Enum):
    """Token types."""
    ACCESS = "access"
    REFRESH = "refresh"
    EMAIL_VERIFICATION = "email_verification"
    PASSWORD_RESET = "password_reset"

# === Configuration ===
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# JWT settings
SECRET_KEY = getattr(settings, "SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable must be set")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = getattr(settings, 'ACCESS_TOKEN_EXPIRE_MINUTES', 30)
REFRESH_TOKEN_EXPIRE_DAYS = getattr(settings, 'REFRESH_TOKEN_EXPIRE_DAYS', 30)
EMAIL_VERIFICATION_EXPIRE_HOURS = getattr(settings, 'EMAIL_VERIFICATION_EXPIRE_HOURS', 24)
PASSWORD_RESET_EXPIRE_HOURS = getattr(settings, 'PASSWORD_RESET_EXPIRE_HOURS', 1)

# === Enhanced Pydantic Models ===
class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    user_id: Optional[int] = None
    token_type: TokenType = TokenType.ACCESS
    scopes: List[str] = []

class UserBase(BaseModel):
    """Base user model."""
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False
    is_verified: bool = False

class UserCreate(UserBase):
    """User creation model."""
    password: str
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class UserUpdate(BaseModel):
    """User update model."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None
    password: Optional[str] = None
    old_password: Optional[str] = None

class UserResponse(UserBase):
    """User response model."""
    id: int
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    auth_provider: AuthProvider = AuthProvider.LOCAL
    
    class Config:
        from_attributes = True

class PasswordReset(BaseModel):
    """Password reset request."""
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    """Password reset confirmation."""
    token: str
    new_password: str
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v

class EmailVerification(BaseModel):
    """Email verification."""
    token: str

class RoleBase(BaseModel):
    """Base role model."""
    name: str
    description: Optional[str] = None

class RoleCreate(RoleBase):
    """Role creation model."""
    permissions: List[str] = []

class RoleResponse(RoleBase):
    """Role response model."""
    id: int
    permissions: List[str] = []
    created_at: datetime
    
    class Config:
        from_attributes = True

# === Enhanced Database Models ===
@register_model_to_admin
class AuthUser(DBBaseModel):
    """Enhanced user database model."""
    __tablename__ = "auth_users"
    
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    auth_provider: Mapped[str] = mapped_column(String(20), default=AuthProvider.LOCAL, nullable=False)
    social_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    failed_login_attempts: Mapped[int] = mapped_column(default=0, nullable=False)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    user_roles = relationship("UserRole", back_populates="user", cascade="all, delete-orphan")

@register_model_to_admin
class RefreshToken(DBBaseModel):
    """Refresh token model."""
    __tablename__ = "refresh_tokens"
    
    token: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("auth_users.id"), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Relationships
    user = relationship("AuthUser", back_populates="refresh_tokens", foreign_keys=[user_id])

@register_model_to_admin
class Role(DBBaseModel):
    """Role model for RBAC."""
    __tablename__ = "roles"
    
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    user_roles = relationship("UserRole", back_populates="role", cascade="all, delete-orphan")
    role_permissions = relationship("RolePermission", back_populates="role", cascade="all, delete-orphan")

@register_model_to_admin
class Permission(DBBaseModel):
    """Permission model."""
    __tablename__ = "permissions"
    
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    role_permissions = relationship("RolePermission", back_populates="permission", cascade="all, delete-orphan")

@register_model_to_admin
class UserRole(DBBaseModel):
    """User-Role association."""
    __tablename__ = "user_roles"
    
    user_id: Mapped[int] = mapped_column(ForeignKey("auth_users.id"), primary_key=True)
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id"), primary_key=True)
    
    # Relationships
    user = relationship("AuthUser", back_populates="user_roles", foreign_keys=[user_id])
    role = relationship("Role", back_populates="user_roles", foreign_keys=[role_id])

@register_model_to_admin
class RolePermission(DBBaseModel):
    """Role-Permission association."""
    __tablename__ = "role_permissions"
    
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id"), primary_key=True)
    permission_id: Mapped[int] = mapped_column(ForeignKey("permissions.id"), primary_key=True)
    
    # Relationships
    role = relationship("Role", back_populates="role_permissions", foreign_keys=[role_id])
    permission = relationship("Permission", back_populates="role_permissions", foreign_keys=[permission_id])

# === Enhanced Utility Functions ===
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return pwd_context.hash(password)

def generate_token() -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(32)

def create_token(data: Dict[str, Any], token_type: TokenType, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT token."""
    to_encode = data.copy()
    to_encode["type"] = token_type.value
    
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        if token_type == TokenType.ACCESS:
            expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        elif token_type == TokenType.REFRESH:
            expire = datetime.now() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        elif token_type == TokenType.EMAIL_VERIFICATION:
            expire = datetime.now() + timedelta(hours=EMAIL_VERIFICATION_EXPIRE_HOURS)
        elif token_type == TokenType.PASSWORD_RESET:
            expire = datetime.now() + timedelta(hours=PASSWORD_RESET_EXPIRE_HOURS)
    
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create an access token."""
    return create_token(data, TokenType.ACCESS, expires_delta)

def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create a refresh token."""
    return create_token(data, TokenType.REFRESH)

def decode_token(token: str, token_type: TokenType = TokenType.ACCESS) -> Dict[str, Any]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != token_type.value:
            raise JWTError("Invalid token type")
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

# === Enhanced Database Operations ===
async def get_user_by_email(db: AsyncSession, email: str) -> Optional[AuthUser]:
    """Get user by email."""
    result = await db.execute(select(AuthUser).where(AuthUser.email == email))
    return result.scalar_one_or_none()

async def get_user_by_username(db: AsyncSession, username: str) -> Optional[AuthUser]:
    """Get user by username."""
    result = await db.execute(select(AuthUser).where(AuthUser.username == username))
    return result.scalar_one_or_none()

async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[AuthUser]:
    """Get user by ID."""
    result = await db.execute(select(AuthUser).where(AuthUser.id == user_id))
    return result.scalar_one_or_none()

async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[AuthUser]:
    """Authenticate user with username/email and password."""
    user = await get_user_by_username(db, username)
    if not user:
        user = await get_user_by_email(db, username)
    
    if not user:
        return None
    
    # Check if account is locked
    if user.locked_until and user.locked_until > datetime.now():
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Account is temporarily locked due to multiple failed login attempts"
        )
    
    if not verify_password(password, user.hashed_password):
        # Increment failed attempts
        user.failed_login_attempts += 1
        if user.failed_login_attempts >= 5:
            user.locked_until = datetime.now() + timedelta(hours=1)
        await db.commit()
        return None
    
    # Reset failed attempts on successful login
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login = datetime.now()
    await db.commit()
    
    return user

async def create_user(db: AsyncSession, user_in: UserCreate) -> AuthUser:
    """Create a new user."""
    hashed_password = get_password_hash(user_in.password)
    user_data = user_in.model_dump(exclude={'password', 'confirm_password'})
    user_data['hashed_password'] = hashed_password
    
    user = AuthUser(**user_data)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

async def store_refresh_token(db: AsyncSession, user_id: int, token: str) -> RefreshToken:
    """Store a refresh token."""
    refresh_token = RefreshToken(
        token=token,
        user_id=user_id,
        expires_at=datetime.now() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )
    db.add(refresh_token)
    await db.commit()
    return refresh_token

async def get_refresh_token(db: AsyncSession, token: str) -> Optional[RefreshToken]:
    """Get refresh token."""
    result = await db.execute(
        select(RefreshToken).where(
            and_(
                RefreshToken.token == token,
                RefreshToken.is_revoked == False,
                RefreshToken.expires_at > datetime.now()
            )
        )
    )
    return result.scalar_one_or_none()

async def revoke_refresh_token(db: AsyncSession, token: str) -> bool:
    """Revoke a refresh token."""
    result = await db.execute(select(RefreshToken).where(RefreshToken.token == token))
    refresh_token = result.scalar_one_or_none()
    if refresh_token:
        refresh_token.is_revoked = True
        await db.commit()
        return True
    return False

# === Role-Based Access Control ===
async def get_user_permissions(db: AsyncSession, user_id: int) -> List[str]:
    """Get all permissions for a user."""
    # This would involve joining user_roles, roles, role_permissions, and permissions
    # Implementation depends on your specific RBAC requirements
    pass

async def user_has_permission(db: AsyncSession, user_id: int, permission: str) -> bool:
    """Check if user has a specific permission."""
    permissions = await get_user_permissions(db, user_id)
    return permission in permissions

# === Enhanced Authentication Dependencies ===
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> AuthUser:
    """Get the current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = decode_token(token, TokenType.ACCESS)
        user_id: int = payload.get("user_id")
        username: str = payload.get("sub")
        if not user_id and not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    if user_id:
        user = await get_user_by_id(db, user_id)
    else:
        user = await get_user_by_username(db, username)
        if not user:
            user = await get_user_by_email(db, username)
    
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: AuthUser = Depends(get_current_user)) -> AuthUser:
    """Get the current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_verified_user(current_user: AuthUser = Depends(get_current_active_user)) -> AuthUser:
    """Get the current verified user."""
    if not current_user.is_verified:
        raise HTTPException(status_code=400, detail="Email not verified")
    return current_user

async def get_current_superuser(current_user: AuthUser = Depends(get_current_active_user)) -> AuthUser:
    """Get current superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

def require_permission(permission: str):
    """Decorator to require specific permission."""
    async def permission_checker(
        current_user: AuthUser = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
    ):
        if current_user.is_superuser:
            return current_user
        
        if not await user_has_permission(db, current_user.id, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    
    return permission_checker

# === Enhanced API Routes ===
router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_in: UserCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user."""
    # Check if user already exists
    if await get_user_by_email(db, user_in.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    if await get_user_by_username(db, user_in.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create new user
    user = await create_user(db, user_in)
    
    # Send verification email (implement this based on your email system)
    # background_tasks.add_task(send_verification_email, user.email, user.id)
    
    return user

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """OAuth2 compatible token login with refresh token."""
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(data={"user_id": user.id})
    await store_refresh_token(db, user.id, refresh_token)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@router.post("/refresh", response_model=Token)
async def refresh_access_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token using refresh token."""
    stored_token = await get_refresh_token(db, refresh_token)
    if not stored_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user = await get_user_by_id(db, stored_token.user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@router.post("/logout")
async def logout(
    refresh_token: str,
    db: AsyncSession = Depends(get_db),
    current_user: AuthUser = Depends(get_current_active_user)
):
    """Logout and revoke refresh token."""
    await revoke_refresh_token(db, refresh_token)
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: AuthUser = Depends(get_current_active_user)):
    """Get current user information."""
    return current_user

@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: AuthUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user information."""
    update_data = user_update.model_dump(exclude_unset=True)
    
    # Verify old password if changing password
    if 'password' in update_data:
        if not user_update.old_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Old password required to change password"
            )
        if not verify_password(user_update.old_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect old password"
            )
        update_data['hashed_password'] = get_password_hash(update_data.pop('password'))
        update_data.pop('old_password', None)
    
    # Check for unique constraints
    if 'email' in update_data and update_data['email'] != current_user.email:
        if await get_user_by_email(db, update_data['email']):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    if 'username' in update_data and update_data['username'] != current_user.username:
        if await get_user_by_username(db, update_data['username']):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
    
    # Update user
    updated_user = await current_user.update(db, **update_data)
    await db.commit()
    return updated_user

@router.post("/request-password-reset")
async def request_password_reset(
    request: PasswordReset,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Request password reset."""
    user = await get_user_by_email(db, request.email)
    if user:
        # Generate reset token
        reset_token = create_token(
            data={"user_id": user.id, "email": user.email},
            token_type=TokenType.PASSWORD_RESET
        )
        # Send reset email (implement based on your email system)
        # background_tasks.add_task(send_password_reset_email, user.email, reset_token)
    
    # Always return success to prevent email enumeration
    return {"message": "If the email exists, a reset link has been sent"}

@router.post("/reset-password")
async def reset_password(
    reset_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db)
):
    """Reset password using token."""
    try:
        payload = decode_token(reset_data.token, TokenType.PASSWORD_RESET)
        user_id = payload.get("user_id")
        email = payload.get("email")
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    user = await get_user_by_id(db, user_id)
    if not user or user.email != email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reset token"
        )
    
    # Update password
    user.hashed_password = get_password_hash(reset_data.new_password)
    user.failed_login_attempts = 0
    user.locked_until = None
    await db.commit()
    
    return {"message": "Password reset successfully"}

@router.post("/verify-email")
async def verify_email(
    verification: EmailVerification,
    db: AsyncSession = Depends(get_db)
):
    """Verify email address."""
    try:
        payload = decode_token(verification.token, TokenType.EMAIL_VERIFICATION)
        user_id = payload.get("user_id")
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )
    
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token"
        )
    
    user.is_verified = True
    await db.commit()
    
    return {"message": "Email verified successfully"}

# Export public API

from .audit import AuthAuditLog, AuditService, AuditAction
from .rate_limiting import rate_limit, InMemoryRateLimiter
from .session_management import UserSession, SessionManager
from .middleware import AuthMiddleware
from .two_factor import TwoFactorAuth, TwoFactorService, two_factor_router
from .social import SocialAuthProvider, GoogleAuthProvider, GitHubAuthProvider, authenticate_social_user
__all__ = [
    'AuthUser', 'UserCreate', 'UserUpdate', 'UserResponse', 'Token',
    'RefreshToken', 'Role', 'Permission', 'UserRole', 'RolePermission',
    'get_current_user', 'get_current_active_user', 'get_current_verified_user',
    'get_current_superuser', 'require_permission',
    'create_access_token', 'create_refresh_token', 'verify_password', 'get_password_hash',
    'authenticate_user', 'create_user', 'get_user_by_email', 'get_user_by_username',
    'router', 'AuthProvider', 'TokenType', 'SocialAuthProvider', 'GoogleAuthProvider', 'GitHubAuthProvider',
    'authenticate_social_user', 'TwoFactorAuth', 'TwoFactorService',
    'two_factor_router', 'AuthAuditLog', 'AuditService', 'AuditAction',
    'rate_limit', 'InMemoryRateLimiter', 'UserSession', 'SessionManager',
    'AuthMiddleware'

]


   