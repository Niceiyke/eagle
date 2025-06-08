# auth/audit.py
"""
Authentication audit logging for Eagle Framework.
"""
import json
from datetime import datetime
from typing import Optional, Dict, Any,List
from enum import Enum
from pydantic import BaseModel
from sqlalchemy import String, DateTime, Text, select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Request
from ..db import BaseModel as DBBaseModel, Mapped, mapped_column
from ..admin import register_model_to_admin

class AuditAction(str, Enum):
    """Audit action types."""
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    REGISTER = "register"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    EMAIL_VERIFICATION = "email_verification"
    TWO_FACTOR_ENABLE = "2fa_enable"
    TWO_FACTOR_DISABLE = "2fa_disable"
    PROFILE_UPDATE = "profile_update"
    ACCOUNT_LOCKED = "account_locked"
    SOCIAL_LOGIN = "social_login"

@register_model_to_admin
class AuthAuditLog(DBBaseModel):
    """Authentication audit log model."""
    __tablename__ = "auth_audit_logs"
    
    user_id: Mapped[Optional[int]] = mapped_column(nullable=True)
    username: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string
    success: Mapped[bool] = mapped_column(default=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

class AuditService:
    """Audit logging service."""
    
    @staticmethod
    async def log_auth_event(
        db: AsyncSession,
        action: AuditAction,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        request: Optional[Request] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log an authentication event."""
        ip_address = None
        user_agent = None
        
        if request:
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
        
        audit_log = AuthAuditLog(
            user_id=user_id,
            username=username,
            action=action.value,
            ip_address=ip_address,
            user_agent=user_agent,
            details=json.dumps(details) if details else None,
            success=success
        )
        
        db.add(audit_log)
        await db.commit()
    
    @staticmethod
    async def get_user_audit_logs(
        db: AsyncSession,
        user_id: int,
        limit: int = 50
    ) -> List[AuthAuditLog]:
        """Get audit logs for a user."""
        result = await db.execute(
            select(AuthAuditLog)
            .where(AuthAuditLog.user_id == user_id)
            .order_by(AuthAuditLog.timestamp.desc())
            .limit(limit)
        )
        return result.scalars().all()