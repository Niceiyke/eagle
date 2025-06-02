from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum as PyEnum
from eagleapi.db import BaseModel
from typing import Optional, Dict, Any
from eagleapi.admin import register_model_to_admin

class LogLevel(PyEnum):
    """Enum for log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@register_model_to_admin
class Log(BaseModel):
    """Model for storing API logs"""
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    level = Column(Enum(LogLevel))
    message = Column(String, nullable=True)
    request_id = Column(String, nullable=True)
    user_id = Column(Integer, nullable=True)
    path = Column(String, nullable=True)
    method = Column(String, nullable=True)
    status_code = Column(Integer, nullable=True)
    request_data = Column(JSON, nullable=True)
    response_data = Column(JSON, nullable=True)
    extra_data = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"Log(id={self.id}, level={self.level}, message={self.message[:50]})"
