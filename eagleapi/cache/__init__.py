# cache/__init__.py
"""
Eagle Caching System - Production-grade caching with minimal configuration.

Supports both in-memory caching (development) and Redis (production) with
automatic fallback and seamless integration.
"""

import asyncio
import json
import pickle
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Dict, List, Callable
from functools import wraps
from datetime import datetime, timedelta
import hashlib
import os

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        pass
    
    @abstractmethod
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache."""
        pass


class InMemoryCache(CacheBackend):
    """In-memory cache backend for development."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._lock = asyncio.Lock()
    
    async def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, data in self._cache.items():
            if data.get('expires_at') and current_time > data['expires_at']:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            self._access_times.pop(key, None)
    
    async def _evict_lru(self):
        """Evict least recently used items if cache is full."""
        if len(self._cache) >= self.max_size:
            # Sort by access time and remove oldest
            sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])
            keys_to_remove = [k for k, _ in sorted_keys[:max(1, len(sorted_keys) // 4)]]
            
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._access_times.pop(key, None)
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            await self._cleanup_expired()
            
            if key not in self._cache:
                return None
            
            data = self._cache[key]
            current_time = time.time()
            
            # Check if expired
            if data.get('expires_at') and current_time > data['expires_at']:
                del self._cache[key]
                self._access_times.pop(key, None)
                return None
            
            # Update access time
            self._access_times[key] = current_time
            return data['value']
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        async with self._lock:
            await self._evict_lru()
            
            expires_at = None
            if ttl or self.default_ttl:
                expires_at = time.time() + (ttl or self.default_ttl)
            
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            self._access_times[key] = time.time()
            return True
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_times.pop(key, None)
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None
    
    async def clear(self) -> bool:
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
            return True
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        for key, value in mapping.items():
            await self.set(key, value, ttl)
        return True


class RedisCache(CacheBackend):
    """Redis cache backend for production."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 default_ttl: int = 300, key_prefix: str = "eagle:"):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self._redis = None
        self._connection_pool = None
    
    async def _get_redis(self):
        """Get Redis connection with lazy initialization."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._connection_pool = redis.ConnectionPool.from_url(
                    self.redis_url, decode_responses=False, max_connections=20
                )
                self._redis = redis.Redis(connection_pool=self._connection_pool)
                await self._redis.ping()
                logger.info("Redis connection established successfully")
            except ImportError:
                logger.error("redis package not installed. Install with: pip install redis")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._redis
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        redis = await self._get_redis()
        try:
            data = await redis.get(self._make_key(key))
            if data is None:
                return None
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        redis = await self._get_redis()
        try:
            data = pickle.dumps(value)
            expire_time = ttl or self.default_ttl
            await redis.setex(self._make_key(key), expire_time, data)
            return True
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        redis = await self._get_redis()
        try:
            result = await redis.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        redis = await self._get_redis()
        try:
            result = await redis.exists(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        redis = await self._get_redis()
        try:
            keys = await redis.keys(f"{self.key_prefix}*")
            if keys:
                await redis.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        redis = await self._get_redis()
        try:
            redis_keys = [self._make_key(key) for key in keys]
            values = await redis.mget(redis_keys)
            result = {}
            for i, value in enumerate(values):
                if value is not None:
                    try:
                        result[keys[i]] = pickle.loads(value)
                    except Exception as e:
                        logger.error(f"Failed to deserialize value for key {keys[i]}: {e}")
            return result
        except Exception as e:
            logger.error(f"Redis get_many error: {e}")
            return {}
    
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        redis = await self._get_redis()
        try:
            pipe = redis.pipeline()
            expire_time = ttl or self.default_ttl
            
            for key, value in mapping.items():
                data = pickle.dumps(value)
                pipe.setex(self._make_key(key), expire_time, data)
            
            await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Redis set_many error: {e}")
            return False


class CacheManager:
    """Main cache manager with automatic backend selection."""
    
    def __init__(self, 
                 backend: Optional[CacheBackend] = None,
                 default_ttl: int = 300,
                 key_prefix: str = "eagle:",
                 redis_url: Optional[str] = None,
                 auto_select_backend: bool = True):
        
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self._backend = backend
        
        if backend is None and auto_select_backend:
            self._backend = self._auto_select_backend(redis_url)
        elif backend is None:
            # Default to in-memory
            self._backend = InMemoryCache(default_ttl=default_ttl)
    
    def _auto_select_backend(self, redis_url: Optional[str] = None) -> CacheBackend:
        """Automatically select the best available backend."""
        # Check environment variables
        from eagleapi.core.config import settings
        redis_url = redis_url or settings.REDIS_URL
        
        # Try Redis first (production)
        if redis_url:
            try:
                return RedisCache(
                    redis_url=redis_url,
                    default_ttl=self.default_ttl,
                    key_prefix=self.key_prefix
                )
            except Exception as e:
                logger.warning(f"Redis backend unavailable, falling back to in-memory: {e}")
        
        # Fallback to in-memory (development)
        logger.debug("Using in-memory cache backend")
        return InMemoryCache(default_ttl=self.default_ttl)
    
    @property
    def backend(self) -> CacheBackend:
        return self._backend
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return await self._backend.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        return await self._backend.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        return await self._backend.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self._backend.exists(key)
    
    async def clear(self) -> bool:
        """Clear all cache."""
        return await self._backend.clear()
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values."""
        return await self._backend.get_many(keys)
    
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values."""
        return await self._backend.set_many(mapping, ttl)
    
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()


# Global cache instance
cache = CacheManager()


def cached(ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_func: Custom function to generate cache key
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{cache.cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to run in async context
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Convenience functions
async def get(key: str) -> Optional[Any]:
    """Get value from cache."""
    return await cache.get(key)


async def set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Set value in cache."""
    return await cache.set(key, value, ttl)


async def delete(key: str) -> bool:
    """Delete key from cache."""
    return await cache.delete(key)


async def clear() -> bool:
    """Clear all cache."""
    return await cache.clear()


# Export main components
__all__ = [
    'CacheManager', 'CacheBackend', 'InMemoryCache', 'RedisCache',
    'cache', 'cached', 'get', 'set', 'delete', 'clear'
]