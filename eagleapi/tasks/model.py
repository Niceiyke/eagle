"""
Task SQLAlchemy model for async background task persistence.
"""
from __future__ import annotations
from typing import Optional, Any, Dict
from datetime import datetime
from sqlalchemy import String, Enum as SAEnum, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column
from eagleapi.db import BaseModel
from eagleapi.admin import register_model_to_admin
import enum

class TaskStatusEnum(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskPriorityEnum(int, enum.Enum):
    LOW = 10
    NORMAL = 5
    HIGH = 2
    CRITICAL = 1

@register_model_to_admin(list_display=["id", "status", "func_name", "priority", "started_at", "finished_at"])
class TaskQueue(BaseModel):
    __tablename__ = "task_queue"

    status: Mapped[TaskStatusEnum] = mapped_column(
        SAEnum(TaskStatusEnum), default=TaskStatusEnum.PENDING, nullable=False, index=True
    )
    priority: Mapped[TaskPriorityEnum] = mapped_column(
        SAEnum(TaskPriorityEnum), default=TaskPriorityEnum.NORMAL, nullable=False, index=True
    )
    func_name: Mapped[str] = mapped_column(String(255), nullable=False)
    args: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    kwargs: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    result: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    def __repr__(self):
        return f"<TaskQueue id={self.id} status={self.status} func={self.func_name}>"
