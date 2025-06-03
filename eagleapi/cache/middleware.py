# cache/middleware.py
"""
Eagle Cache Middleware for HTTP response caching.
"""

import time
import hashlib
import json
from typing import Optional, List, Callable, Any, Dict, Tuple
from fastapi import Request, Response
from starlette.responses import Response as StarletteResponse, StreamingResponse
import logging

from ..middleware.base import EagleMiddleware
from . import cache

logger = logging.getLogger(__name__)


class CacheMiddleware(EagleMiddleware):
    """
    Middleware for caching HTTP responses.
    
    This middleware caches responses based on request method, path, query parameters,
    and specified headers. It integrates with the Eagle middleware system.
    """
    
    def __init__(self, 
                 app,
                 default_ttl: int = 300,
                 cache_methods: List[str] = None,
                 cache_status_codes: List[int] = None,
                 excluded_paths: List[str] = None,
                 key_builder: Optional[Callable] = None,
                 vary_headers: List[str] = None):
        
        self.default_ttl = default_ttl
        self.cache_methods = cache_methods or ['GET']
        self.cache_status_codes = cache_status_codes or [200]
        self.excluded_paths = excluded_paths or []
        self.key_builder = key_builder or self._default_key_builder
        self.vary_headers = vary_headers or []
        
        # Store cached response during request processing
        self._cached_response = None
        
        super().__init__(app, 
                          default_ttl=default_ttl,
                          cache_methods=cache_methods,
                          cache_status_codes=cache_status_codes,
                          excluded_paths=excluded_paths,
                          vary_headers=vary_headers)
    
    def _default_key_builder(self, request: Request) -> str:
        """Default cache key builder.
        
        Generates a unique key based on:
        - HTTP method
        - Full URL path
        - Query parameters
        - Path parameters
        - Vary headers
        - Request body (for non-GET requests)
        """
        # Get the full URL components
        url = request.url
        
        # Get path parameters
        path_params = {}
        if hasattr(request, 'path_params'):
            path_params = request.path_params
            
        # Get query parameters
        query_params = {}
        if url.query:
            query_params = dict(request.query_params)
            
        # Sort query params for consistent key generation
        sorted_query = '&'.join(f"{k}={v}" for k, v in sorted(query_params.items()))
        
        key_parts = [
            request.method.upper(),
            str(url.path),
            f"query:{sorted_query}",
            f"params:{str(sorted(path_params.items()))}"
        ]
        
        # Add vary headers to key
        for header in self.vary_headers:
            header_value = request.headers.get(header, '')
            key_parts.append(f"{header}:{header_value}")
        
        # Include request body for non-GET requests to prevent cache poisoning
        if request.method.upper() != 'GET':
            try:
                body = request._body or b''
                if body:
                    key_parts.append(f"body_hash:{hashlib.md5(body).hexdigest()}")
            except Exception as e:
                logger.warning(f"Error including request body in cache key: {e}")
        
        # Create a stable key
        key_data = '|'.join(str(part) for part in key_parts if part is not None)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        
        logger.debug(f"Generated cache key: {key_hash} for {request.method} {url.path}")
        logger.debug(f"Key parts: {key_parts}")
        return f"http_cache:{key_hash}"
    
    def _should_cache(self, request: Request) -> bool:
        """Determine if request should be cached."""
        # Check method
        if request.method not in self.cache_methods:
            return False
        
        # Check excluded paths
        path = request.url.path
        return not any(path.startswith(excluded) for excluded in self.excluded_paths)
    
    async def before_request(self, request: Request) -> None:
        """Check cache and return cached response if available."""
        self._cached_response = None  # Reset cached response
        
        if not self._should_cache(request):
            logger.debug(f"Skipping cache for {request.method} {request.url.path} - Request not cacheable")
            return
            
        cache_key = self._default_key_builder(request)
        request.state.cache_key = cache_key
        
        logger.debug(f"Checking cache for key: {cache_key}")
        logger.debug(f"Request URL: {request.url}")
        logger.debug(f"Query params: {dict(request.query_params)}")
        logger.debug(f"Path params: {getattr(request, 'path_params', {})}")
        
        try:
            cached = await cache.get(cache_key)
            if cached:
                logger.info(f"Cache HIT for {request.method} {request.url.path}")
                logger.debug(f"Cached data: {cached}")
                self._cached_response = self._create_response_from_cache(cached)
                if self._cached_response:
                    self._cached_response.headers['X-Cache'] = 'HIT'
                    self._cached_response.headers['X-Cache-Key'] = cache_key
                return
                
            logger.debug(f"Cache MISS for {request.method} {request.url.path}")
        except Exception as e:
            logger.error(f"Cache get error: {e}", exc_info=True)
    
    def _create_response_from_cache(self, cached: dict) -> Response:
        """Create a Response object from cached data."""
        try:
            if not isinstance(cached, dict):
                logger.error(f"Invalid cache data type: {type(cached)}")
                return None
                
            # Ensure required fields exist
            required_fields = ['body', 'status_code', 'headers']
            for field in required_fields:
                if field not in cached:
                    logger.error(f"Missing required field in cache: {field}")
                    return None
            
            # Create a copy of headers to avoid modifying the cached ones
            headers = dict(cached['headers'])
            
            # Add cache hit headers
            headers['X-Cache'] = 'HIT'
            
            logger.debug(f"Creating response from cache with status {cached['status_code']}")
            
            return Response(
                content=cached['body'],
                status_code=cached['status_code'],
                headers=headers,
                media_type=cached.get('media_type')
            )
        except Exception as e:
            logger.error(f"Error creating response from cache: {e}", exc_info=True)
            return None
    
    async def after_response(self, request: Request, response: Response) -> Response:
        """Cache the response if appropriate."""
        # Return cached response if available
        if self._cached_response:
            logger.debug("Returning cached response")
            return self._cached_response
            
        # Check if we should cache this response
        if not self._should_cache(request):
            logger.debug("Not caching - Request not eligible")
            return response
            
        if not hasattr(request.state, 'cache_key'):
            logger.debug("Not caching - No cache key in request state")
            return response
            
        if response.status_code not in self.cache_status_codes:
            logger.debug(f"Not caching - Status code {response.status_code} not in cacheable status codes")
            return response
            
        # Cache the response
        cache_key = request.state.cache_key
        logger.debug(f"Attempting to cache response for key: {cache_key}")
        
        try:
            # Handle different response types
            response_body = b""
            response_headers = dict(response.headers)
            response_media_type = getattr(response, 'media_type', None)
            
            if hasattr(response, 'body_iterator'):
                # Handle streaming responses
                logger.debug("Processing streaming response")
                chunks = []
                async for chunk in response.body_iterator:
                    chunks.append(chunk)
                response_body = b"".join(chunks)
            elif hasattr(response, 'body') and response.body is not None:
                response_body = response.body
            else:
                logger.debug("Response has no body to cache")
                return response
            
            # Prepare cache data
            cache_data = {
                'body': response_body,
                'status_code': response.status_code,
                'headers': response_headers,
                'media_type': response_media_type
            }
            
            # Store in cache
            logger.debug(f"Caching response with status {response.status_code}")
            await cache.set(cache_key, cache_data, ttl=self.default_ttl)
            logger.info(f"Cached response for {request.method} {request.url.path}")
            
            # Return a new response with cache headers
            response_headers['X-Cache'] = 'MISS'
            response_headers['X-Cache-Key'] = cache_key
            
            logger.debug(f"Returning response with cache headers: {dict(response_headers)}")
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=response_headers,
                media_type=response_media_type
            )
            
        except Exception as e:
            logger.error(f"Cache set error: {e}", exc_info=True)
            return response


# Export all components
__all__ = [
    'CacheMiddleware', 'CacheStats', 'CacheInvalidator',
    'cache_result', 'invalidate_cache', 'warm_cache', 'cache_aside'
]