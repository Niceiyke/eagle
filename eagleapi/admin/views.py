from eagleapi.db.models.logs import Log, LogLevel
from eagleapi.admin import ModelRegistry
from eagleapi.admin import register_model_to_admin
from eagleapi.utils import format_datetime
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from eagleapi.core.config import settings
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException

@ModelRegistry.register
@ModelRegistry.register_model_to_admin(name="Logs")
class LogAdmin:
    """Admin view for managing logs"""
    
    model = Log
    name = "Logs"
    icon = "fas fa-file-alt"
    
    list_display = [
        "id",
        "timestamp",
        "level",
        "message",
        "user",
        "path",
        "status_code",
    ]
    
    list_filters = [
        "level",
        "timestamp",
        "user",
        "path",
        "status_code",
    ]
    
    search_fields = [
        "message",
        "path",
        "request_data",
        "response_data",
    ]
    
    @classmethod
    async def get_list(cls, db: AsyncSession, skip: int = 0, limit: int = 100,
                      filters: Optional[dict] = None) -> List[Log]:
        """Get list of logs with filtering"""
        query = select(Log)
        
        if filters:
            if "level" in filters:
                query = query.filter(Log.level == LogLevel(filters["level"]))
            if "timestamp" in filters:
                query = query.filter(Log.timestamp >= filters["timestamp"])
            if "user" in filters:
                query = query.filter(Log.user_id == filters["user"])
            if "path" in filters:
                query = query.filter(Log.path.like(f"%{filters["path"]}%"))
            if "status_code" in filters:
                query = query.filter(Log.status_code == filters["status_code"])
        
        result = await db.execute(query.order_by(Log.timestamp.desc()).offset(skip).limit(limit))
        return result.scalars().all()
    
    @classmethod
    async def get_detail(cls, db: AsyncSession, id: int) -> Log:
        """Get detailed log information"""
        result = await db.execute(select(Log).filter_by(id=id))
        log = result.scalar_one_or_none()
        if not log:
            raise HTTPException(status_code=404, detail="Log not found")
        return log
    
    @classmethod
    async def get_filters(cls, db: AsyncSession) -> dict:
        """Get filter options for the logs list"""
        return {
            "level": [level.value for level in LogLevel],
            "status_code": [200, 400, 401, 403, 404, 500],
            "timestamp": [
                ("Last 24 hours", datetime.utcnow() - timedelta(days=1)),
                ("Last 7 days", datetime.utcnow() - timedelta(days=7)),
                ("Last 30 days", datetime.utcnow() - timedelta(days=30)),
            ]
        }
    
    @classmethod
    async def format_list_item(cls, log: Log) -> dict:
        """Format log item for list display"""
        return {
            "id": log.id,
            "timestamp": format_datetime(log.timestamp),
            "level": log.level.value.upper(),
            "message": log.message,
            "user": log.user.username if log.user else "-",
            "path": log.path,
            "status_code": log.status_code,
        }
