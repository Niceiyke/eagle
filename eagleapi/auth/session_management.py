# auth/session_management.py
"""
Advanced session management for Eagle Framework.
"""
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel
from sqlalchemy import String, DateTime, Boolean, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Request, HTTPException, status
from ..db import BaseModel as DBBaseModel, Mapped, mapped_column
from ..admin import register_model_to_admin

@register_model_to_admin
class UserSession(DBBaseModel):
    """User session model."""
    __tablename__ = "user_sessions"
    
    user_id: Mapped[int] = mapped_column(nullable=False)
    session_token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    last_activity: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

class SessionInfo(BaseModel):
    """Session information model."""
    id: int
    ip_address: Optional[str]
    user_agent: Optional[str]
    location: Optional[str]
    is_current: bool
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    
    class Config:
        from_attributes = True

class SessionManager:
    """Session management service."""
    
    @staticmethod
    async def create_session(
        db: AsyncSession,
        user_id: int,
        session_token: str,
        request: Request,
        expires_in_days: int = 30
    ) -> UserSession:
        """Create a new user session."""
        session = UserSession(
            user_id=user_id,
            session_token=session_token,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days)
        )
        
        db.add(session)
        await db.commit()
        return session
    
    @staticmethod
    async def get_user_sessions(
        db: AsyncSession,
        user_id: int,
        current_session_token: Optional[str] = None
    ) -> List[SessionInfo]:
        """Get all active sessions for a user."""
        result = await db.execute(
            select(UserSession)
            .where(
                and_(
                    UserSession.user_id == user_id,
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.utcnow()
                )
            )
            .order_by(UserSession.last_activity.desc())
        )
        
        sessions = result.scalars().all()
        session_infos = []
        
        for session in sessions:
            session_info = SessionInfo(
                id=session.id,
                ip_address=session.ip_address,
                user_agent=session.user_agent,
                location=session.location,
                is_current=session.session_token == current_session_token,
                created_at=session.created_at,
                last_activity=session.last_activity,
                expires_at=session.expires_at
            )
            session_infos.append(session_info)
        
        return session_infos
    
    @staticmethod
    async def revoke_session(
        db: AsyncSession,
        session_id: int,
        user_id: int
    ) -> bool:
        """Revoke a specific session."""
        result = await db.execute(
            select(UserSession)
            .where(
                and_(
                    UserSession.id == session_id,
                    UserSession.user_id == user_id
                )
            )
        )
        
        session = result.scalar_one_or_none()
        if session:
            session.is_active = False
            await db.commit()
            return True
        
        return False
    
    @staticmethod
    async def revoke_all_sessions(
        db: AsyncSession,
        user_id: int,
        except_session_token: Optional[str] = None
    ):
        """Revoke all sessions for a user except the current one."""
        query = select(UserSession).where(UserSession.user_id == user_id)
        
        if except_session_token:
            query = query.where(UserSession.session_token != except_session_token)
        
        result = await db.execute(query)
        sessions = result.scalars().all()
        
        for session in sessions:
            session.is_active = False
        
        await db.commit()
    
    @staticmethod
    async def update_session_activity(
        db: AsyncSession,
        session_token: str
    ):
        """Update session last activity."""
        result = await db.execute(
            select(UserSession)
            .where(UserSession.session_token == session_token)
        )
        
        session = result.scalar_one_or_none()
        if session:
            session.last_activity = datetime.utcnow()
            await db.commit()
