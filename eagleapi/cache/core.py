# cache/core.py
"""Core caching functionality and manager."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Union, Callable
from contextlib import asynccontextmanager
import weakref
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Cache statistics tracking."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_time: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    def __init__(self, **config):
        self.config = config
        self.stats = CacheStats()
        self._lock = asyncio.Lock()
    
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
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def ttl(self, key: str) -> Optional[int]:
        """Get TTL for key."""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        pass
    
    @abstractmethod
    async def close(self):
        """Close backend connections."""
        pass
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats

class Serializer(ABC):
    """Abstract base class for cache serializers."""
    
    @abstractmethod
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes."""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data."""
        pass

class Cache:
    """Main cache interface providing high-level caching operations."""
    
    def __init__(
        self,
        backend: CacheBackend,
        serializer: Optional[Serializer] = None,
        key_prefix: str = "eagle:",
        default_ttl: Optional[int] = 3600,
        enable_stats: bool = True
    ):
        self.backend = backend
        self.serializer = serializer
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.enable_stats = enable_stats
        self._circuit_breaker = CircuitBreaker()
        
    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix."""
        return f"{self.key_prefix}{key}"
    
    @asynccontextmanager
    async def _with_stats(self, operation: str):
        """Context manager for tracking operation stats."""
        start_time = time.time()
        try:
            yield
            if self.enable_stats:
                if operation == 'get':
                    self.backend.stats.hits += 1
                elif operation == 'set':
                    self.backend.stats.sets += 1
                elif operation == 'delete':
                    self.backend.stats.deletes += 1
        except Exception as e:
            if self.enable_stats:
                self.backend.stats.errors += 1
                if operation == 'get':
                    self.backend.stats.misses += 1
            raise
        finally:
            if self.enable_stats:
                self.backend.stats.total_time += time.time() - start_time
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        if not await self._circuit_breaker.can_execute():
            return default
            
        cache_key = self._make_key(key)
        
        try:
            async with self._with_stats('get'):
                value = await self.backend.get(cache_key)
                if value is None:
                    return default
                
                if self.serializer:
                    return self.serializer.deserialize(value)
                return value
                
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            await self._circuit_breaker.record_failure()
            return default
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        if not await self._circuit_breaker.can_execute():
            return False
            
        cache_key = self._make_key(key)
        ttl = ttl if ttl is not None else self.default_ttl
        
        try:
            async with self._with_stats('set'):
                if self.serializer:
                    value = self.serializer.serialize(value)
                
                result = await self.backend.set(cache_key, value, ttl)
                await self._circuit_breaker.record_success()
                return result
                
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            await self._circuit_breaker.record_failure()
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not await self._circuit_breaker.can_execute():
            return False
            
        cache_key = self._make_key(key)
        
        try:
            async with self._with_stats('delete'):
                result = await self.backend.delete(cache_key)
                await self._circuit_breaker.record_success()
                return result
                
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            await self._circuit_breaker.record_failure()
            return False
    
    async def get_or_set(
        self, 
        key: str, 
        factory: Callable,
        ttl: Optional[int] = None
    ) -> Any:
        """Get value from cache or set it using factory function."""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Generate value using factory
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()
        
        await self.set(key, value, ttl)
        return value
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        results = {}
        tasks = []
        
        for key in keys:
            task = asyncio.create_task(self.get(key))
            tasks.append((key, task))
        
        for key, task in tasks:
            try:
                value = await task
                if value is not None:
                    results[key] = value
            except Exception as e:
                logger.warning(f"Error getting key {key}: {e}")
        
        return results
    
    async def set_many(
        self, 
        mapping: Dict[str, Any], 
        ttl: Optional[int] = None
    ) -> Dict[str, bool]:
        """Set multiple values in cache."""
        tasks = []
        
        for key, value in mapping.items():
            task = asyncio.create_task(self.set(key, value, ttl))
            tasks.append((key, task))
        
        results = {}
        for key, task in tasks:
            try:
                results[key] = await task
            except Exception as e:
                logger.warning(f"Error setting key {key}: {e}")
                results[key] = False
        
        return results
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        cache_pattern = self._make_key(pattern)
        try:
            keys = await self.backend.keys(cache_pattern)
            if not keys:
                return 0
            
            tasks = [self.backend.delete(key) for key in keys]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return sum(1 for result in results if result is True)
        except Exception as e:
            logger.warning(f"Error invalidating pattern {pattern}: {e}")
            return 0
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            return await self.backend.clear()
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
            return False
    
    async def stats(self) -> CacheStats:
        """Get cache statistics."""
        return await self.backend.get_stats()

class CircuitBreaker:
    """Circuit breaker for cache operations."""
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Exception = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()
    
    async def can_execute(self) -> bool:
        """Check if operation can be executed."""
        async with self._lock:
            if self.state == 'CLOSED':
                return True
            elif self.state == 'OPEN':
                if (time.time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    async def record_success(self):
        """Record successful operation."""
        async with self._lock:
            self.failure_count = 0
            self.state = 'CLOSED'
    
    async def record_failure(self):
        """Record failed operation."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'

class CacheManager:
    """Global cache manager for handling multiple cache instances."""
    
    def __init__(self):
        self._caches: Dict[str, Cache] = {}
        self._default_cache: Optional[Cache] = None
        self._lock = threading.Lock()
    
    def register_cache(self, name: str, cache: Cache, is_default: bool = False):
        """Register a cache instance."""
        with self._lock:
            self._caches[name] = cache
            if is_default or self._default_cache is None:
                self._default_cache = cache
    
    def get_cache(self, name: Optional[str] = None) -> Optional[Cache]:
        """Get cache instance by name or default."""
        with self._lock:
            if name is None:
                return self._default_cache
            return self._caches.get(name)
    
    def remove_cache(self, name: str):
        """Remove cache instance."""
        with self._lock:
            if name in self._caches:
                cache = self._caches.pop(name)
                if cache == self._default_cache:
                    self._default_cache = next(iter(self._caches.values()), None)
    
    async def close_all(self):
        """Close all cache backends."""
        with self._lock:
            caches = list(self._caches.values())
        
        for cache in caches:
            try:
                await cache.backend.close()
            except Exception as e:
                logger.warning(f"Error closing cache backend: {e}")
    
    def list_caches(self) -> List[str]:
        """List all registered cache names."""
        with self._lock:
            return list(self._caches.keys())

# Global cache manager instance
cache_manager = CacheManager()