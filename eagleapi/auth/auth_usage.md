# Authentication Audit Logging Usage Guide

This document provides examples of how to use the `audit.py` module to track authentication events in your Eagle Framework application.

## Basic Setup

First, ensure you have an instance of `AsyncSession` from SQLAlchemy available in your route handlers or services.

```python
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, Request
from .audit import AuditService, AuditAction

# Example FastAPI dependency to get database session
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
```

## Example 1: Logging User Login

```python
from fastapi import APIRouter, Request, Depends

router = APIRouter()

@router.post("/login")
async def login(
    request: Request,
    credentials: LoginSchema,
    db: AsyncSession = Depends(get_db)
):
    user = await authenticate_user(credentials.email, credentials.password)
    if not user:
        # Log failed login attempt
        await AuditService.log_auth_event(
            db=db,
            action=AuditAction.LOGIN_FAILED,
            username=credentials.email,
            request=request,
            success=False,
            details={"reason": "Invalid credentials"}
        )
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    # Log successful login
    await AuditService.log_auth_event(
        db=db,
        action=AuditAction.LOGIN,
        user_id=user.id,
        username=user.email,
        request=request
    )
    
    return {"message": "Login successful"}
```

## Example 2: Logging User Registration

```python
@router.post("/register")
async def register(
    request: Request,
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    # Check if user exists
    existing_user = await get_user_by_email(db, user_data.email)
    if existing_user:
        await AuditService.log_auth_event(
            db=db,
            action=AuditAction.REGISTER,
            username=user_data.email,
            request=request,
            success=False,
            details={"reason": "Email already registered"}
        )
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = await create_user(db, user_data)
    
    # Log successful registration
    await AuditService.log_auth_event(
        db=db,
        action=AuditAction.REGISTER,
        user_id=user.id,
        username=user.email,
        request=request,
        details={"registration_method": "email"}
    )
    
    return {"message": "User registered successfully"}
```

## Example 3: Logging Password Changes

```python
@router.post("/change-password")
async def change_password(
    request: Request,
    password_data: PasswordChangeSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        await AuditService.log_auth_event(
            db=db,
            action=AuditAction.PASSWORD_CHANGE,
            user_id=current_user.id,
            username=current_user.email,
            request=request,
            success=False,
            details={"reason": "Incorrect current password"}
        )
        raise HTTPException(status_code=400, detail="Incorrect current password")
    
    # Update password
    await update_user_password(db, current_user.id, password_data.new_password)
    
    # Log successful password change
    await AuditService.log_auth_event(
        db=db,
        action=AuditAction.PASSWORD_CHANGE,
        user_id=current_user.id,
        username=current_user.email,
        request=request
    )
    
    return {"message": "Password updated successfully"}
```

## Example 4: Logging 2FA Events

```python
@router.post("/enable-2fa")
async def enable_2fa(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        # Enable 2FA logic here
        secret = await enable_user_2fa(db, current_user.id)
        
        # Log successful 2FA enablement
        await AuditService.log_auth_event(
            db=db,
            action=AuditAction.TWO_FACTOR_ENABLE,
            user_id=current_user.id,
            username=current_user.email,
            request=request
        )
        
        return {"secret": secret}
    except Exception as e:
        await AuditService.log_auth_event(
            db=db,
            action=AuditAction.TWO_FACTOR_ENABLE,
            user_id=current_user.id,
            username=current_user.email,
            request=request,
            success=False,
            details={"error": str(e)}
        )
        raise HTTPException(status_code=400, detail=str(e))
```

## Example 5: Retrieving Audit Logs

```python
@router.get("/audit-logs")
async def get_my_audit_logs(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50
):
    logs = await AuditService.get_user_audit_logs(
        db=db,
        user_id=current_user.id,
        limit=min(limit, 100)  # Limit to 100 records max
    )
    
    return [
        {
            "action": log.action,
            "timestamp": log.timestamp.isoformat(),
            "ip_address": log.ip_address,
            "success": log.success,
            "details": json.loads(log.details) if log.details else None
        }
        for log in logs
    ]
```

## Available Audit Actions

The `AuditAction` enum provides the following action types:

- `LOGIN`: Successful user login
- `LOGOUT`: User logout
- `LOGIN_FAILED`: Failed login attempt
- `REGISTER`: New user registration
- `PASSWORD_CHANGE`: Password change
- `PASSWORD_RESET`: Password reset
- `EMAIL_VERIFICATION`: Email verification
- `2FA_ENABLE`: Two-factor authentication enabled
- `2FA_DISABLE`: Two-factor authentication disabled
- `PROFILE_UPDATE`: User profile updated
- `ACCOUNT_LOCKED`: User account locked
- `SOCIAL_LOGIN`: Social media login

## Best Practices

1. Always include the `request` object when available to capture IP and user agent
2. Provide meaningful details in the `details` dictionary for debugging
3. Set `success=False` for failed operations
4. Use appropriate action types from `AuditAction`
5. Consider rate limiting or batching for high-volume logging

## Viewing Logs in Admin

Audit logs are automatically registered with the admin interface (if using `eagle-admin`). You can view and search logs through the admin panel at `/admin/auth_audit_logs/`.
