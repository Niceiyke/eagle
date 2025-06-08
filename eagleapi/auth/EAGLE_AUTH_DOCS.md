# Eagle Auth Documentation

## Overview

Eagle Auth is an advanced authentication and authorization module for the Eagle Framework. It provides:

- JWT authentication (access & refresh tokens)
- OAuth2 and social login (Google, GitHub, Facebook)
- Role-Based Access Control (RBAC)
- Two-Factor Authentication (2FA)
- Rate limiting, audit logging, and session management

---

## 1. Installation & Setup

Make sure you have Eagle Framework and its dependencies installed. Add the following to your requirements:

```bash
pip install fastapi sqlalchemy passlib[bcrypt] python-jose pyotp qrcode httpx
```

---

## 2. Quickstart Example

### Register Eagle Auth in your FastAPI app

```python
from fastapi import FastAPI
from eagleapi.auth import router as auth_router, AuthMiddleware

app = FastAPI()

# Register the authentication routes
app.include_router(auth_router)

# Optionally add authentication middleware
app.add_middleware(AuthMiddleware)
```

---

## 3. Authentication Features

### User Registration

```python
POST /auth/register
{
  "username": "yourname",
  "email": "your@email.com",
  "password": "yourpassword",
  "confirm_password": "yourpassword"
}
```

### Login (JWT)

```python
POST /auth/token
{
  "username": "yourname or email",
  "password": "yourpassword"
}
```
Response includes `access_token`, `refresh_token`, and expiry.

### Refresh Token

```python
POST /auth/refresh
{
  "refresh_token": "<your_refresh_token>"
}
```

### Get Current User

```python
GET /auth/me
Authorization: Bearer <access_token>
```

---

## 4. Social Authentication

Supported providers: Google, GitHub, Facebook.

#### Example: Google OAuth2

1. Set credentials in your `.env`:
    ```
    GOOGLE_CLIENT_ID=your_client_id
    GOOGLE_CLIENT_SECRET=your_secret
    GOOGLE_REDIRECT_URI=https://yourdomain.com/auth/google/callback
    ```

2. Redirect user to `/auth/google/login` with required params.

3. On callback, exchange the code for tokens:
    ```python
    from eagleapi.auth import authenticate_social_user, AuthProvider

    user = await authenticate_social_user(AuthProvider.GOOGLE, code, db)
    ```

---

## 5. Two-Factor Authentication (2FA)

Enable extra security for users.

- **Setup:**  
  `POST /auth/2fa/setup`  
  Returns secret, QR code, and backup codes.

- **Verify:**  
  `POST /auth/2fa/verify`  
  Submit the code from your authenticator app.

- **Enable:**  
  `POST /auth/2fa/enable`  
  Confirm and enable 2FA.

- **Disable:**  
  `POST /auth/2fa/disable`  
  Provide password to disable 2FA.

---

## 6. Role-Based Access Control (RBAC)

Define roles and permissions for fine-grained access control.

- Assign roles to users.
- Protect endpoints using the `require_permission` decorator.

```python
from eagleapi.auth import require_permission

@app.get("/admin")
@require_permission("admin_access")
async def admin_dashboard():
    ...
```

---

## 7. Session Management

Track user sessions, enforce session policies, and manage active sessions.

- Sessions are automatically managed for each login.
- Use `SessionManager` utilities for advanced control.

---

## 8. Audit Logging

All critical auth events (login, logout, registration, 2FA, etc.) are logged for security and compliance.

- Access logs via `AuditService`.

---

## 9. Rate Limiting

Protect your API from abuse by applying rate limits.

```python
from eagleapi.auth import rate_limit

@app.get("/sensitive-endpoint")
@rate_limit(limit=5, period=60)  # 5 requests per minute
async def sensitive():
    ...
```

---

## 10. Extending Eagle Auth

You can extend or override models, routers, and services to fit your application's needs.

- Add new providers by subclassing `SocialAuthProvider`.
- Customize user models by extending `AuthUser`.

---

## 11. Import Reference

You can import the following from `eagleapi.auth`:

```python
from eagleapi.auth import (
    AuthUser, UserCreate, UserUpdate, UserResponse, Token,
    get_current_user, get_current_active_user, get_current_superuser,
    require_permission, create_access_token, create_refresh_token,
    authenticate_user, create_user, get_user_by_email, get_user_by_username,
    router, AuthProvider, TokenType, SocialAuthProvider, GoogleAuthProvider, GitHubAuthProvider,
    authenticate_social_user, TwoFactorAuth, TwoFactorService, two_factor_router,
    AuthAuditLog, AuditService, AuditAction, rate_limit, InMemoryRateLimiter,
    UserSession, SessionManager, AuthMiddleware
)
```

---

## 12. Best Practices

- Always use HTTPS in production.
- Store secrets and credentials securely (e.g., environment variables).
- Regularly audit roles, permissions, and user activity.
- Enable 2FA for sensitive accounts.

---

## 13. Further Reading

- See `auth_usage.md` and `social_auth_usage.md` in the `eagleapi/auth` directory for advanced examples and integration details.
