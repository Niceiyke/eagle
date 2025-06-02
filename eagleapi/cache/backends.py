# cache/backends.py
"""Cache backend implementations."""

import asyncio
import time
import json
import pickle
from typing import Any, Optional, List, Dict
from .core import CacheBackend

class InMemoryBackend(CacheBackend):
    """In-memory cache backend using dictionaries."""
    
    def __init__(self, max_size: int = 1000, cleanup_interval: int = 60, **config):
        super().__init__(**config)
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self._data: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
        self._access_times: Dict[str, float] = {}
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired())
    
    async def _cleanup_expired(self):
        """Background task to clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                current_time = time.time()
                
                expired_keys = [
                    key for key, exp_time in self._expiry.items()
                    if exp_time <= current_time
                ]
                
                for key in expired_keys:
                    self._data.pop(key, None)
                    self._expiry.pop(key, None)
                    self._access_times.pop(key, None)
                
                # LRU eviction if over max_size
                if len(self._data) > self.max_size:
                    # Sort by access time and remove oldest
                    sorted_keys = sorted(
                        self._access_times.items(), 
                        key=lambda x: x[1]
                    )
                    keys_to_remove = sorted_keys[:len(self._data) - self.max_size]
                    
                    for key, _ in keys_to_remove:
                        self._data.pop(key, None)
                        self._expiry.pop(key, None)
                        self._access_times.pop(key, None)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in cleanup task: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from in-memory cache."""
        current_time = time.time()
        
        # Check if key exists and hasn't expired
        if key in self._data:
            if key in self._expiry and self._expiry[key] <= current_time:
                # Key has expired, remove it
                self._data.pop(key, None)
                self._expiry.pop(key, None)
                self._access_times.pop(key, None)
                return None
            
            # Update access time for LRU
            self._access_times[key] = current_time
            return self._data[key]
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in in-memory cache."""
        current_time = time.time()
        
        self._data[key] = value
        self._access_times[key] = current_time
        
        if ttl is not None:
            self._expiry[key] = current_time + ttl
        elif key in self._expiry:
            # Remove expiry if ttl is None
            self._expiry.pop(key, None)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from in-memory cache."""
        if key in self._data:
            self._data.pop(key, None)
            self._expiry.pop(key, None)
            self._access_times.pop(key, None)
            return True
        return False
    
    async def clear(self) -> bool:
        """Clear all entries from in-memory cache."""
        self._data.clear()
        self._expiry.clear()
        self._access_times.clear()
        return True
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return await self.get(key) is not None
    
    async def ttl(self, key: str) -> Optional[int]:
        """Get TTL for key."""
        if key not in self._data:
            return None
        
        if key not in self._expiry:
            return -1  # No expiration
        
        remaining = self._expiry[key] - time.time()
        return int(remaining) if remaining > 0 else 0
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        import fnmatch
        return [key for key in self._data.keys() if fnmatch.fnmatch(key, pattern)]
    
    async def close(self):
        """Close backend and cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

class RedisBackend(CacheBackend):
    """Redis cache backend."""
    
    def __init__(self, url: str = "redis://localhost:6379", **config):
        super().__init__(**config)
        self.url = url
        self.redis = None
        self._pool = None
    
    async def _get_redis(self):
        """Get Redis connection."""
        if self.redis is None:
            try:
                import aioredis
                self.redis = aioredis.from_url(self.url)
                await self.redis.ping()
            except ImportError:
                raise ImportError("aioredis is required for Redis backend. Install with: pip install aioredis")
            except Exception as e:
                raise Exception(f"Failed to connect to Redis: {e}")
        return self.redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        redis = await self._get_redis()
        value = await redis.get(key)
        return value if value is not None else None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        redis = await self._get_redis()
        try:
            if ttl is not None:
                await redis.setex(key, ttl, value)
            else:
                await redis.set(key, value)
            return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        redis = await self._get_redis()
        result = await redis.delete(key)
        return result > 0
    
    async def clear(self) -> bool:
        """Clear all entries from Redis."""
        redis = await self._get_redis()
        try:
            await redis.flushdb()
            return True
        except Exception:
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        redis = await self._get_redis()
        result = await redis.exists(key)
        return result > 0
    
    async def ttl(self, key: str) -> Optional[int]:
        """Get TTL for key in Redis."""
        redis = await self._get_redis()
        result = await redis.ttl(key)
        return result if result >= 0 else None
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern from Redis."""
        redis = await self._get_redis()
        keys = await redis.keys(pattern)
        return [key.decode() if isinstance(key, bytes) else key for key in keys]
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()

class MemcachedBackend(CacheBackend):
    """Memcached cache backend."""
    
    def __init__(self, servers: List[str] = None, **config):
        super().__init__(**config)
        self.servers = servers or ['127.0.0.1:11211']
        self.client = None
    
    async def _get_client(self):
        """Get Memcached client."""
        if self.client is None:
            try:
                import aiomcache
                self.client = aiomcache.Client(*self.servers)
            except ImportError:
                raise ImportError("aiomcache is required for Memcached backend. Install with: pip install aiomcache")
        return self.client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Memcached."""
        client = await self._get_client()
        try:
            value = await client.get(key.encode())
            return value if value is not None else None
        except Exception:
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Memcached."""
        client = await self._get_client()
        try:
            exptime = ttl if ttl is not None else 0
            await client.set(key.encode(), value, exptime=exptime)
            return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Memcached."""
        client = await self._get_client()
        try:
            await client.delete(key.encode())
            return True
        except Exception:
            return False
    
    async def clear(self) -> bool:
        """Clear all entries from Memcached."""
        client = await self._get_client()
        try:
            await client.flush_all()
            return True
        except Exception:
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Memcached."""
        value = await self.get(key)
        return value is not None
    
    async def ttl(self, key: str) -> Optional[int]:
        """Get TTL for key - Memcached doesn't support TTL queries."""
        return None  # Memcached doesn't support TTL queries
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern - Memcached doesn't support key listing."""
        return []  # Memcached doesn't support key enumeration
    
    async def close(self):
        """Close Memcached connection."""
        if self.client:
            await self.client.close()
