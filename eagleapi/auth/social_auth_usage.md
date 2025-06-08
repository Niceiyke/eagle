# Social Authentication and Session Management

This guide provides comprehensive examples for using the social authentication and session management features in the Eagle Framework.

## Table of Contents
1. [Social Authentication](#social-authentication)
2. [Session Management](#session-management)
3. [Core Authentication Features](#core-authentication-features)

## Social Authentication

### 1. Google OAuth2 Setup

```python
# In your .env file
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=https://yourdomain.com/auth/google/callback
```

### 2. GitHub OAuth2 Setup

```python
# In your .env file
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
GITHUB_REDIRECT_URI=https://yourdomain.com/auth/github/callback
```

### 3. Social Login Endpoint Example

```python
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from . import (
    AuthProvider,
    authenticate_social_user,
    create_access_token,
    create_refresh_token,
    get_db,
    Token
)

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.get("/{provider}/login")
async def social_login(
    provider: str,
    code: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    try:
        # Convert provider string to AuthProvider enum
        auth_provider = AuthProvider(provider.lower())
        
        # Authenticate user with the social provider
        user = await authenticate_social_user(auth_provider, code, db)
        
        # Create tokens
        access_token = create_access_token(data={"sub": user.email, "user_id": user.id})
        refresh_token = create_refresh_token(data={"sub": user.email, "user_id": user.id})
        
        # Store refresh token in database
        await store_refresh_token(db, user.id, refresh_token)
        
        # In a real app, you might want to set these as HTTP-only cookies
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider: {provider}. Supported providers are: {', '.join(p.value for p in AuthProvider if p != AuthProvider.LOCAL)}"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during social login: {str(e)}"
        )
```

## Session Management

### 1. Getting Active Sessions

```python
@router.get("/sessions", response_model=List[SessionInfo])
async def get_my_sessions(
    current_user: AuthUser = Depends(get_current_active_user),
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all active sessions for the current user.
    """
    return await SessionManager.get_user_sessions(
        db=db,
        user_id=current_user.id,
        current_session_token=token
    )
```

### 2. Revoking Sessions

```python
@router.delete("/sessions/{session_id}")
async def revoke_session(
    session_id: int,
    current_user: AuthUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Revoke a specific session.
    """
    session = await db.get(UserSession, session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to revoke this session")
    
    session.is_active = False
    await db.commit()
    
    return {"message": "Session revoked successfully"}

@router.post("/sessions/revoke-all")
async def revoke_all_other_sessions(
    current_user: AuthUser = Depends(get_current_active_user),
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
):
    """
    Revoke all other active sessions except the current one.
    """
    result = await db.execute(
        update(UserSession)
        .where(
            and_(
                UserSession.user_id == current_user.id,
                UserSession.is_active == True,
                UserSession.session_token != token
            )
        )
        .values(is_active=False)
    )
    
    await db.commit()
    return {"message": f"Revoked {result.rowcount} sessions"}
```

## Core Authentication Features

### 1. Role-Based Access Control (RBAC)

```python
# Create a protected route that requires admin role
@router.get("/admin/dashboard")
@require_permission("admin:read")
async def admin_dashboard(
    current_user: AuthUser = Depends(get_current_active_user)
):
    return {"message": "Welcome to the admin dashboard"}

# Create a route that requires specific permission
@router.get("/reports")
@require_permission("reports:view")
async def view_reports(
    current_user: AuthUser = Depends(get_current_active_user)
):
    return {"message": "Viewing reports"}
```

### 2. Two-Factor Authentication (2FA)

```python
@router.post("/2fa/enable")
async def enable_2fa(
    current_user: AuthUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Enable 2FA for the current user.
    Returns a QR code URL for setting up an authenticator app.
    """
    if current_user.totp_secret:
        raise HTTPException(status_code=400, detail="2FA is already enabled")
    
    # Generate a secret
    secret = pyotp.random_base32()
    
    # Create a provisioning URI for the authenticator app
    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=current_user.email,
        issuer_name="Your App Name"
    )
    
    # Generate a QR code URL (requires qrcode library)
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(totp_uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64 for the API response
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    qr_code_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Store the secret (temporarily, until verified)
    current_user.totp_secret = secret
    await db.commit()
    
    return {
        "secret": secret,
        "qr_code": f"data:image/png;base64,{qr_code_base64}",
        "manual_entry_code": secret
    }

@router.post("/2fa/verify")
async def verify_2fa(
    code: str,
    current_user: AuthUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Verify 2FA setup with a code from the authenticator app.
    """
    if not current_user.totp_secret:
        raise HTTPException(status_code=400, detail="2FA is not set up")
    
    totp = pyotp.TOTP(current_user.totp_secret)
    if not totp.verify(code):
        raise HTTPException(status_code=400, detail="Invalid verification code")
    
    # Mark 2FA as verified
    current_user.is_2fa_enabled = True
    await db.commit()
    
    return {"message": "2FA has been enabled successfully"}
```

### 3. Rate Limiting

```python
from ..auth.rate_limiting import rate_limit

@router.get("/protected-route")
@rate_limit(limit=5, window=60)  # 5 requests per minute
async def protected_route():
    return {"message": "This is a rate-limited route"}
```

## Best Practices

1. **Security**
   - Always use HTTPS in production
   - Store sensitive configuration in environment variables
   - Implement proper CORS policies
   - Use secure, HTTP-only cookies for tokens when possible

2. **Performance**
   - Cache frequently accessed user data
   - Use background tasks for non-critical operations (e.g., sending emails)
   - Implement proper database indexing for user-related queries

3. **User Experience**
   - Provide clear error messages
   - Implement proper session management
   - Support multiple authentication methods
   - Make it easy for users to manage their sessions

4. **Monitoring**
   - Log authentication events
   - Monitor for suspicious activity
   - Set up alerts for multiple failed login attempts

This guide covers the main aspects of social authentication and session management in the Eagle Framework. For more advanced use cases, refer to the official documentation or the source code.
