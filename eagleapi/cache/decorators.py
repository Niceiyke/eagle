# cache/decorators.py
"""Cache decorators for Eagle API."""

import asyncio
import hashlib
import inspect
from functools import wraps
from typing import Any, Optional, Callable, Union, List
from .core import cache_manager

def cache_key(*args, prefix: str = "", separator: str = ":") -> Callable:
    """
    Generate cache key from function arguments.
    
    Args:
        *args: Arguments to include in key generation
        prefix: Optional prefix for the key
        separator: Separator between key components
    
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*call_args, **call_kwargs):
            # Build key components
            key_parts = [prefix] if prefix else []
            key_parts.append(func.__name__)
            
            # Add specified args
            if args:
                for arg_name in args:
                    if arg_name in call_kwargs:
                        key_parts.append(str(call_kwargs[arg_name]))
                    else:
                        # Try to get positional arg by name
                        sig = inspect.signature(func)
                        param_names = list(sig.parameters.keys())
                        if arg_name in param_names:
                            arg_index = param_names.index(arg_name)
                            if arg_index < len(call_args):
                                key_parts.append(str(call_args[arg_index]))
            else:
                # Include all args if none specified
                key_parts.extend([str(arg) for arg in call_args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(call_kwargs.items())])
            
            return separator.join(key_parts)
        
        return wrapper
    return decorator

def cached(
    ttl: Optional[int] = None,
    key: Optional[Union[str, Callable]] = None,
    cache_name: Optional[str] = None,
    skip_cache: Optional[Callable] = None,
    on_cache_hit: Optional[Callable] = None,
    on_cache_miss: Optional[Callable] = None
):
    """
    Cache decorator for functions and methods.
    
    Args:
        ttl: Time to live in seconds
        key: Cache key or function to generate key
        cache_name: Name of cache instance to use
        skip_cache: Function to determine if cache should be skipped
        on_cache_hit: Callback for cache hits
        on_cache_miss: Callback for cache misses
    
    Example:
        @cached(ttl=300, key="user:{user_id}")
        async def get_user(user_id: int):
            return await db.get_user(user_id)
        
        @cached(ttl=60)
        async def expensive_calculation(x: int, y: int):
            return x ** y
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = cache_manager.get_cache(cache_name)
            if not cache:
                # No cache available, execute function
                return await func(*args, **kwargs)
            
            # Check if we should skip cache
            if skip_cache and skip_cache(*args, **kwargs):
                return await func(*args, **kwargs)
            
            # Generate cache key
            if key is None:
                # Auto-generate key from function name and args
                key_parts = [func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                cache_key = ":".join(key_parts)
            elif callable(key):
                cache_key = key(*args, **kwargs)
            else:
                # String key with format support
                cache_key = key.format(*args, **kwargs)
            
            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                if on_cache_hit:
                    on_cache_hit(cache_key, cached_value)
                return cached_value
            
            # Cache miss - execute function
            if on_cache_miss:
                on_cache_miss(cache_key)
            
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(cache_key, result, ttl)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to handle async cache operations
            cache = cache_manager.get_cache(cache_name)
            if not cache:
                return func(*args, **kwargs)
            
            # Check if we should skip cache
            if skip_cache and skip_cache(*args, **kwargs):
                return func(*args, **kwargs)
            
            # Generate cache key
            if key is None:
                # Auto-generate key from function name and args
                key_parts = [func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                cache_key = ":".join(key_parts)
            elif callable(key):
                cache_key = key(*args, **kwargs)
            else:
                # String key with format support
                cache_key = key.format(*args, **kwargs)
            
            # Try to get from cache synchronously
            try:
                loop = asyncio.get_event_loop()
                cached_value = loop.run_until_complete(cache.get(cache_key))
                if cached_value is not None:
                    if on_cache_hit:
                        on_cache_hit(cache_key, cached_value)
                    return cached_value
            except Exception:
                # If async operations fail, just execute function
                pass
            
            # Cache miss - execute function
            if on_cache_miss:
                on_cache_miss(cache_key)
            
            result = func(*args, **kwargs)
            
            # Store in cache
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(cache.set(cache_key, result, ttl))
            except Exception:
                # Cache set failed, but we have the result
                pass
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def invalidate_cache(
    pattern: Optional[str] = None,
    keys: Optional[List[str]] = None,
    cache_name: Optional[str] = None
):
    """
    Decorator to invalidate cache entries after function execution.
    
    Args:
        pattern: Pattern to match keys for invalidation
        keys: Specific keys to invalidate
        cache_name: Name of cache instance to use
    
    Example:
        @invalidate_cache(pattern="user:*")
        async def update_user(user_id: int, data: dict):
            return await db.update_user(user_id, data)
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            cache = cache_manager.get_cache(cache_name)
            if cache:
                try:
                    if pattern:
                        await cache.invalidate_pattern(pattern)
                    if keys:
                        for key in keys:
                            await cache.delete(key)
                except Exception as e:
                    logger.warning(f"Cache invalidation failed: {e}")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            cache = cache_manager.get_cache(cache_name)
            if cache:
                try:
                    loop = asyncio.get_event_loop()
                    if pattern:
                        loop.run_until_complete(cache.invalidate_pattern(pattern))
                    if keys:
                        for key in keys:
                            loop.run_until_complete(cache.delete(key))
                except Exception as e:
                    logger.warning(f"Cache invalidation failed: {e}")
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator