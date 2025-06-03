# cache/decorators.py
"""
Advanced caching decorators for Eagle framework.
"""

from functools import wraps
from typing import Optional, Callable, Any, List
import asyncio
import hashlib
import inspect

from .import cache
from .utils import CacheInvalidator


def cache_result(ttl: Optional[int] = None, 
                key_prefix: str = "",
                key_func: Optional[Callable] = None,
                tags: Optional[List[str]] = None):
    """
    Enhanced caching decorator with tagging support.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        key_func: Custom key generation function
        tags: List of tags for cache invalidation
    """
    def decorator(func):
        func_name = f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                key_data = f"{func_name}:{str(bound_args.arguments)}"
                cache_key = f"{key_prefix}{hashlib.md5(key_data.encode()).hexdigest()}"
            
            # Try cache first
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result, ttl)
            
            # Store tags if provided
            if tags:
                for tag in tags:
                    tag_key = f"tag:{tag}"
                    tagged_keys = await cache.get(tag_key) or []
                    if cache_key not in tagged_keys:
                        tagged_keys.append(cache_key)
                        await cache.set(tag_key, tagged_keys)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def invalidate_cache(*tags: str):
    """Decorator to invalidate cache tags after function execution."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidate tags
            from .utils import CacheInvalidator
            await CacheInvalidator.invalidate_tags(list(tags))
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            asyncio.run(CacheInvalidator.invalidate_tags(list(tags)))
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator
