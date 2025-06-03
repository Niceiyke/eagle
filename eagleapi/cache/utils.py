# cache/utils.py
"""
Cache utilities and helpers.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from functools import wraps

from .import cache


class CacheStats:
    """Cache statistics tracker."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
        self.start_time = datetime.now()
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'deletes': self.deletes,
            'errors': self.errors,
            'hit_rate': self.hit_rate,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }


class CacheInvalidator:
    """Utility for cache invalidation patterns."""
    
    @staticmethod
    async def invalidate_pattern(pattern: str):
        """Invalidate cache keys matching a pattern."""
        # This is a simple implementation
        # In production, you might want to use Redis SCAN or similar
        logger.warning("Pattern invalidation not fully implemented for all backends")
    
    @staticmethod
    async def invalidate_tags(tags: List[str]):
        """Invalidate cache entries with specific tags."""
        for tag in tags:
            tag_key = f"tag:{tag}"
            tagged_keys = await cache.get(tag_key)
            if tagged_keys:
                for key in tagged_keys:
                    await cache.delete(key)
                await cache.delete(tag_key)


async def warm_cache(key: str, func: callable, *args, **kwargs):
    """Warm cache with function result."""
    result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
    await cache.set(key, result)
    return result


def cache_aside(key: str, func: callable, ttl: Optional[int] = None):
    """Cache-aside pattern implementation."""
    async def get_or_compute(*args, **kwargs):
        # Try cache first
        cached_value = await cache.get(key)
        if cached_value is not None:
            return cached_value
        
        # Compute and cache
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        await cache.set(key, result, ttl)
        return result
    
    return get_or_compute