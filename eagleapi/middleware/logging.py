# middleware/logging.py
from fastapi.responses import Response, JSONResponse
from fastapi.requests import Request
from sqlalchemy.ext.asyncio import AsyncSession
from eagleapi.db import get_db
from eagleapi.db.models.logs import Log, LogLevel
from .base import EagleMiddleware
import time

from typing import Optional, Set, List, Dict, Any
import json
import logging
import re
import asyncio
from contextlib import asynccontextmanager

logger = logging.getLogger("eagle.middleware.logging")

class LoggingMiddleware(EagleMiddleware):
    """Enhanced logging middleware with improved performance and features."""
    
    def setup(self):
        self.log_methods = self.config.get('log_methods')
        self.log_status_codes = self.config.get('log_status_codes')
        self.log_paths = self.config.get('log_paths', [])
        self.excluded_paths = self.config.get('excluded_paths', [])
        self.log_request_body = self.config.get('log_request_body', True)
        self.log_response_body = self.config.get('log_response_body', False)
        self.max_body_size = self.config.get('max_body_size', 10 * 1024)
        self.batch_logging = self.config.get('batch_logging', False)
        
        # Compile regex patterns
        self._path_patterns = self._compile_patterns(self.log_paths)
        self._excluded_patterns = self._compile_patterns(self.excluded_paths)
        
        if self.batch_logging:
            self._log_batch = []
            self._batch_lock = asyncio.Lock()
    
    def _extract_request_data(self, request: Request) -> dict:
        """Extract relevant data from the request for logging."""
        try:
            request_data = {
                'method': request.method,
                'url': str(request.url),
                'headers': dict(request.headers),
                'query_params': dict(request.query_params),
            }
            
            # Add request body if available and within size limit
            if hasattr(request.state, 'cached_body'):
                try:
                    body = request.state.cached_body
                    if body:
                        request_data['body'] = body.decode('utf-8', errors='replace')
                except Exception as e:
                    logger.warning(f"Error reading request body: {e}")
            
            return request_data
            
        except Exception as e:
            logger.warning(f"Error extracting request data: {e}")
            return {'error': str(e)}
    
    def _compile_patterns(self, paths: List[str]) -> List[re.Pattern]:
        """Compile regex patterns for path matching."""
        patterns = []
        for path in paths:
            try:
                if any(char in path for char in r'.*+?{}[]|()'):
                    patterns.append(re.compile(path))
                else:
                    patterns.append(re.compile(re.escape(path)))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{path}': {e}")
        return patterns
    
    def should_log_request(self, method: str, path: str) -> bool:
        """Determine if request should be logged."""
        if self.log_methods and method not in self.log_methods:
            return False
        
        # Check exclusions first
        for pattern in self._excluded_patterns:
            if pattern.match(path):
                return False
        
        # If no specific paths, log everything not excluded
        if not self._path_patterns:
            return True
        
        # Check if path matches inclusion patterns
        return any(pattern.match(path) for pattern in self._path_patterns)
    
    async def before_request(self, request: Request):
        """Process request before handling."""
        if not self.should_log_request(request.method, request.url.path):
            request.state.skip_logging = True
            return
        
        request.state.skip_logging = False
        
        # Cache request body if needed
        if self.log_request_body:
            try:
                body = await request.body()
                if len(body) <= self.max_body_size:
                    request.state.cached_body = body
            except Exception as e:
                logger.warning(f"Error caching request body: {e}")
    
    async def after_response(self, request: Request, response: Response) -> Response:
        """Log request/response after processing."""
        if getattr(request.state, 'skip_logging', True):
            return response
        
        # Log asynchronously to avoid blocking
        asyncio.create_task(self._log_async(request, response))
        return response
    
    async def handle_exception(self, request: Request, exc: Exception) -> Response:
        """Handle and log exceptions."""
        if not getattr(request.state, 'skip_logging', True):
            asyncio.create_task(self._log_error_async(request, exc))
        
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )
    
    async def _log_async(self, request: Request, response: Response):
        """Asynchronously log request and response."""
        try:
            async with self._get_db_session() as db:
                if db:
                    await self._create_log_entries(request, response, db)
        except Exception as e:
            logger.error(f"Logging error: {e}")
    
    async def _log_error_async(self, request: Request, exc: Exception):
        """Asynchronously log errors."""
        try:
            async with self._get_db_session() as db:
                if db:
                    await self._create_error_log(request, exc, db)
        except Exception as e:
            logger.error(f"Error logging failed: {e}")
    
    @asynccontextmanager
    async def _get_db_session(self):
        """Get database session with proper error handling for middleware."""
        session = None
        try:
            # Use the get_db dependency function directly
            async for db_session in get_db():
                session = db_session
                break
            
            if session:
                yield session
                # Let the middleware handle commits explicitly
            else:
                yield None
                
        except Exception as e:
            logger.warning(f"DB session error: {e}")
            if session and session.in_transaction():
                try:
                    await session.rollback()
                except Exception:
                    pass
            yield None
    
    async def _create_log_entries(self, request: Request, response: Response, db: AsyncSession):
        """Create log entries for request and response."""
        request_id = request.state.request_id
        processing_time = time.time() - request.state.start_time
        
        # Log request
        log = Log(
            level=LogLevel.INFO,
            message=f"Request: {request.method} {request.url.path}",
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            request_data=self._extract_request_data(request),
            extra_data={
                "ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "processing_time": processing_time,
            }
        )
        db.add(log)
        
        # Log response if configured
        if response.status_code >= 400 or self.log_response_body:
            response_log = Log(
                level=LogLevel.ERROR if response.status_code >= 400 else LogLevel.INFO,
                message=f"Response: {request.method} {request.url.path} - {response.status_code}",
                request_id=request_id,
                path=request.url.path,
                method=request.method,
                status_code=response.status_code,
                extra_data={"processing_time": processing_time}
            )
            db.add(response_log)

        await db.commit()