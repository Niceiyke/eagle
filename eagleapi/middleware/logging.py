from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession
from eagleapi.db import get_db
from eagleapi.db.models.logs import Log, LogLevel
from typing import Callable, Optional, Set, List, Union
from datetime import datetime
import uuid
import json
import logging
import re

logger = logging.getLogger("eagle.middleware")

class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        log_methods: Optional[Set[str]] = None,
        log_status_codes: Optional[Set[int]] = None,
        log_paths: Optional[List[str]] = None,
        log_request_body: bool = True,
        log_response_body: bool = False,
    ):
        super().__init__(app)
        self.log_methods = {m.upper() for m in log_methods} if log_methods else None
        self.log_status_codes = log_status_codes
        self.log_paths = log_paths or []
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body

        self._compiled_path_patterns = [
            re.compile(path) if path.startswith("^") else path for path in self.log_paths
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = datetime.utcnow()

        method = request.method.upper()
        path = request.url.path

        should_log = self.should_log_request(method, path)
        if not should_log:
            return await call_next(request)

        body = await request.body()
        request._body = body

        try:
            db: AsyncSession = await anext(get_db())
        except Exception as e:
            logger.warning(f"DB unavailable: {e}")
            return await call_next(request)

        try:
            await self.log_request(request, request_id, db)

            response = await call_next(request)

            if self.should_log_response(response.status_code):
                await self.log_response(request, response, request_id, db)

            await db.commit()
            return response

        except Exception as e:
            await db.rollback()
            await self.log_error(request, e, request_id, db)
            raise

    def should_log_request(self, method: str, path: str) -> bool:
        if self.log_methods and method not in self.log_methods:
            return False
        if not self._compiled_path_patterns:
            return True
        for pattern in self._compiled_path_patterns:
            if isinstance(pattern, str) and pattern == path:
                return True
            if hasattr(pattern, "match") and pattern.match(path):
                return True
        return False

    def should_log_response(self, status_code: int) -> bool:
        if self.log_status_codes is None:
            return True
        return status_code in self.log_status_codes

    async def log_request(self, request: Request, request_id: str, db: AsyncSession) -> None:
        data = self.extract_json_safely(request._body) if self.log_request_body else None
        log = Log(
            level=LogLevel.INFO,
            message=f"Request: {request.method} {request.url.path}",
            request_id=request_id,
            user_id=self.get_user_id(request),
            path=request.url.path,
            method=request.method,
            request_data=data,
            extra_data={"ip": request.client.host if request.client else None}
        )
        db.add(log)

    async def log_response(self, request: Request, response: Response, request_id: str, db: AsyncSession) -> None:
        data = await self.extract_response_body(response) if self.log_response_body else None
        log = Log(
            level=LogLevel.INFO,
            message=f"Response: {request.method} {request.url.path} - {response.status_code}",
            request_id=request_id,
            user_id=self.get_user_id(request),
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            response_data=data,
            extra_data={
                "processing_time": (datetime.utcnow() - request.state.start_time).total_seconds()
            }
        )
        db.add(log)

    async def log_error(self, request: Request, error: Exception, request_id: str, db: AsyncSession) -> None:
        log = Log(
            level=LogLevel.ERROR,
            message=f"Error: {request.method} {request.url.path} - {str(error)}",
            request_id=request_id,
            user_id=self.get_user_id(request),
            path=request.url.path,
            method=request.method,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            extra_data={
                "error_type": type(error).__name__,
                "detail": str(error),
            }
        )
        db.add(log)
        try:
            await db.commit()
        except Exception as commit_error:
            logger.error(f"Error committing log: {commit_error}")
            await db.rollback()

    def extract_json_safely(self, body: bytes) -> Optional[dict]:
        try:
            return json.loads(body.decode()) if body else None
        except Exception:
            return None

    async def extract_response_body(self, response: Response) -> Optional[dict]:
        try:
            body = await response.body()
            return json.loads(body.decode()) if body else None
        except Exception:
            return None

    def get_user_id(self, request: Request) -> Optional[int]:
        try:
            return getattr(request.user, "id", None)
        except Exception:
            return None
