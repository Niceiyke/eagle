from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession
from eagleapi.db import get_db
from eagleapi.db.models.logs import Log, LogLevel
from typing import Callable, Optional, Set, List, Union, Dict, Any
from datetime import datetime
import uuid
import json
import logging
import re
import asyncio
from contextlib import asynccontextmanager
import time

logger = logging.getLogger("eagle.middleware")

class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        log_methods: Optional[Set[str]] = None,
        log_status_codes: Optional[Set[int]] = None,
        log_paths: Optional[List[str]] = None,
        excluded_paths: Optional[List[str]] = None,
        log_request_body: bool = True,
        log_response_body: bool = False,
        max_body_size: int = 10 * 1024,  # 10KB limit
        enable_performance_logging: bool = True,
        batch_logging: bool = False,
        batch_size: int = 10,
        batch_timeout: float = 5.0,
    ):
        super().__init__(app)
        self.log_methods = {m.upper() for m in log_methods} if log_methods else None
        self.log_status_codes = log_status_codes
        self.log_paths = log_paths or []
        self.excluded_paths = excluded_paths or []
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size
        self.enable_performance_logging = enable_performance_logging
        self.batch_logging = batch_logging
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        # Compile regex patterns once during initialization
        self._compiled_path_patterns = self._compile_patterns(self.log_paths)
        self._compiled_excluded_patterns = self._compile_patterns(self.excluded_paths)
        
        # Batch logging support
        if self.batch_logging:
            self._log_batch: List[Log] = []
            self._batch_lock = asyncio.Lock()
            self._last_batch_time = time.time()

    def _compile_patterns(self, paths: List[str]) -> List[Union[str, re.Pattern]]:
        """Compile regex patterns once during initialization."""
        patterns = []
        for path in paths:
            if path.startswith("^") or any(char in path for char in r'.*+?{}[]|()'):
                try:
                    patterns.append(re.compile(path))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{path}': {e}")
                    patterns.append(path)  # Fallback to string matching
            else:
                patterns.append(path)
        return patterns

    @asynccontextmanager
    async def get_db_session(self):
        """Context manager for database session handling."""
        db = None
        try:
            # Get a new session
            db = await anext(get_db())
            try:
                yield db
                # If we get here, the block completed successfully
                await db.commit()
            except Exception as e:
                # If there was an error, rollback
                await db.rollback()
                raise
            finally:
                # Don't close the session here, let the dependency injection handle it
                # This prevents the session from being closed while still in use
                pass
        except Exception as e:
            logger.warning(f"DB error: {e}")
            # If we couldn't get a session, yield None
            yield None
        finally:
            # We're not closing the session here anymore to prevent race conditions
            # The session will be closed by the dependency injection system
            pass

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID and set timing
        request_id = str(uuid.uuid4())
        start_time = time.time()
        request.state.request_id = request_id
        request.state.start_time = start_time

        method = request.method.upper()
        path = request.url.path

        # Early return if we shouldn't log this request
        if not self.should_log_request(method, path):
            return await call_next(request)

        # Read and cache request body if needed
        request_body = None
        if self.log_request_body:
            try:
                body_bytes = await request.body()
                if len(body_bytes) <= self.max_body_size:
                    request_body = body_bytes
                    # Cache the body for the actual request handler
                    request._body = body_bytes
                else:
                    logger.debug(f"Request body too large ({len(body_bytes)} bytes), skipping")
            except Exception as e:
                logger.warning(f"Error reading request body: {e}")

        # Process request
        response = None
        error = None
        
        try:
            response = await call_next(request)
        except Exception as e:
            error = e
            # Create a generic error response
            response = Response(
                content=json.dumps({"error": "Internal server error"}),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                media_type="application/json"
            )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Log asynchronously to avoid blocking the response
        asyncio.create_task(self._log_request_response(
            request, response, request_id, request_body, processing_time, error
        ))

        if error:
            raise error

        return response

    def should_log_request(self, method: str, path: str) -> bool:
        """Determine if request should be logged with improved logic."""
        # Check if method should be logged
        if self.log_methods and method not in self.log_methods:
            return False
        
        # Check excluded paths first (more efficient)
        if self._path_matches_patterns(path, self._compiled_excluded_patterns):
            return False
        
        # If no specific paths to log, log everything (except excluded)
        if not self._compiled_path_patterns:
            return True
        
        # Check if path matches inclusion patterns
        return self._path_matches_patterns(path, self._compiled_path_patterns)

    def _path_matches_patterns(self, path: str, patterns: List[Union[str, re.Pattern]]) -> bool:
        """Check if path matches any of the compiled patterns."""
        for pattern in patterns:
            if isinstance(pattern, str):
                if pattern == path:
                    return True
            else:  # regex pattern
                if pattern.match(path):
                    return True
        return False

    def should_log_response(self, status_code: int) -> bool:
        """Determine if response should be logged."""
        if self.log_status_codes is None:
            return True
        return status_code in self.log_status_codes

    async def _log_request_response(
        self, 
        request: Request, 
        response: Response, 
        request_id: str, 
        request_body: Optional[bytes], 
        processing_time: float,
        error: Optional[Exception] = None
    ) -> None:
        """Log request and response asynchronously."""
        async with self.get_db_session() as db:
            if db is None:
                return

            try:
                # Log request
                await self._create_request_log(request, request_id, request_body, db)
                
                # Log response or error
                if error:
                    await self._create_error_log(request, error, request_id, processing_time, db)
                elif self.should_log_response(response.status_code):
                    await self._create_response_log(request, response, request_id, processing_time, db)

                if self.batch_logging:
                    await self._handle_batch_logging(db)
                else:
                    await db.commit()

            except Exception as e:
                logger.error(f"Error during logging: {e}")
                try:
                    await db.rollback()
                except Exception:
                    pass

    async def _create_request_log(
        self, 
        request: Request, 
        request_id: str, 
        request_body: Optional[bytes], 
        db: AsyncSession
    ) -> None:
        """Create request log entry."""
        request_data = self._extract_json_safely(request_body) if request_body else None
        
        extra_data = {
            "ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
        }
        
        # Add query parameters if present
        if request.url.query:
            extra_data["query_params"] = str(request.url.query)

        log = Log(
            level=LogLevel.INFO,
            message=f"Request: {request.method} {request.url.path}",
            request_id=request_id,
            user_id=self._get_user_id(request),
            path=request.url.path,
            method=request.method,
            request_data=request_data,
            extra_data=extra_data
        )
        
        if self.batch_logging:
            await self._add_to_batch(log)
        else:
            db.add(log)

    async def _create_response_log(
        self, 
        request: Request, 
        response: Response, 
        request_id: str, 
        processing_time: float, 
        db: AsyncSession
    ) -> None:
        """Create response log entry."""
        response_data = None
        if self.log_response_body:
            response_data = await self._extract_response_body_safe(response)

        extra_data = {
            "processing_time": processing_time,
            "content_type": response.headers.get("content-type"),
        }

        if self.enable_performance_logging:
            # Add performance metrics
            if processing_time > 1.0:  # Log slow requests
                extra_data["performance_warning"] = "slow_request"

        log = Log(
            level=LogLevel.INFO,
            message=f"Response: {request.method} {request.url.path} - {response.status_code}",
            request_id=request_id,
            user_id=self._get_user_id(request),
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            response_data=response_data,
            extra_data=extra_data
        )
        
        if self.batch_logging:
            await self._add_to_batch(log)
        else:
            db.add(log)

    async def _create_error_log(
        self, 
        request: Request, 
        error: Exception, 
        request_id: str, 
        processing_time: float, 
        db: AsyncSession
    ) -> None:
        """Create error log entry."""
        log = Log(
            level=LogLevel.ERROR,
            message=f"Error: {request.method} {request.url.path} - {str(error)}",
            request_id=request_id,
            user_id=self._get_user_id(request),
            path=request.url.path,
            method=request.method,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            extra_data={
                "error_type": type(error).__name__,
                "detail": str(error)[:1000],  # Truncate long error messages
                "processing_time": processing_time,
            }
        )
        
        if self.batch_logging:
            await self._add_to_batch(log)
        else:
            db.add(log)
            # Immediately commit error logs
            try:
                await db.commit()
            except Exception as commit_error:
                logger.error(f"Error committing error log: {commit_error}")
                await db.rollback()

    async def _add_to_batch(self, log: Log) -> None:
        """Add log to batch for batch processing."""
        async with self._batch_lock:
            self._log_batch.append(log)

    async def _handle_batch_logging(self, db: AsyncSession) -> None:
        """Handle batch logging logic."""
        should_flush = False
        
        async with self._batch_lock:
            current_time = time.time()
            if (len(self._log_batch) >= self.batch_size or 
                current_time - self._last_batch_time >= self.batch_timeout):
                should_flush = True

        if should_flush:
            await self._flush_batch(db)

    async def _flush_batch(self, db: AsyncSession) -> None:
        """Flush the current batch of logs to database."""
        async with self._batch_lock:
            if not self._log_batch:
                return
            
            batch_to_process = self._log_batch.copy()
            self._log_batch.clear()
            self._last_batch_time = time.time()

        try:
            db.add_all(batch_to_process)
            await db.commit()
            logger.debug(f"Flushed {len(batch_to_process)} logs to database")
        except Exception as e:
            logger.error(f"Error flushing log batch: {e}")
            await db.rollback()

    def _extract_json_safely(self, body: bytes) -> Optional[Dict[str, Any]]:
        """Safely extract JSON from request body."""
        if not body:
            return None
        
        try:
            text = body.decode('utf-8')
            return json.loads(text)
        except UnicodeDecodeError:
            # Try with different encoding or treat as binary
            try:
                text = body.decode('latin-1')
                return {"_raw_body_size": len(body), "_encoding": "latin-1"}
            except Exception:
                return {"_raw_body_size": len(body), "_encoding": "binary"}
        except json.JSONDecodeError:
            # Return first 200 chars if not JSON
            try:
                text = body.decode('utf-8')[:200]
                return {"_raw_body_preview": text, "_body_size": len(body)}
            except Exception:
                return {"_raw_body_size": len(body)}
        except Exception as e:
            logger.debug(f"Error extracting request body: {e}")
            return None

    async def _extract_response_body_safe(self, response: Response) -> Optional[Dict[str, Any]]:
        """Safely extract response body without consuming the stream."""
        try:
            # Handle different response types
            if isinstance(response, StreamingResponse):
                # Don't try to read streaming responses
                return {"_type": "streaming_response"}
            
            if hasattr(response, 'body'):
                body = response.body
                if isinstance(body, bytes):
                    if len(body) > self.max_body_size:
                        return {"_body_size": len(body), "_truncated": True}
                    
                    try:
                        return json.loads(body.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        return {"_raw_body_preview": body.decode('utf-8', errors='ignore')[:200]}
            
            return None
        except Exception as e:
            logger.debug(f"Error extracting response body: {e}")
            return None

    def _get_user_id(self, request: Request) -> Optional[int]:
        """Safely extract user ID from request."""
        try:
            # Try multiple ways to get user ID
            if hasattr(request, 'user') and request.user:
                if hasattr(request.user, 'id'):
                    return request.user.id
                elif hasattr(request.user, 'user_id'):
                    return request.user.user_id
            
            # Try from state
            if hasattr(request.state, 'user_id'):
                return request.state.user_id
            
            return None
        except Exception:
            return None