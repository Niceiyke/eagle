# cache/decorators.py
"""
Advanced caching decorators for Eagle framework with proper serialization.
"""

from functools import wraps
from typing import Optional, Callable, Any, List
import asyncio
import hashlib
import inspect
import logging
import json
from datetime import datetime, date
from decimal import Decimal

from . import cache

logger = logging.getLogger(__name__)

class CacheSerializer:
    """Handles serialization and deserialization of complex objects for caching."""
    
    @staticmethod
    def serialize(obj: Any) -> Any:
        """
        Serialize object for caching.
        
        Args:
            obj: The object to serialize
            
        Returns:
            The serialized object or None if serialization fails
        """
        logger.debug(f"Serializing object of type: {type(obj).__name__}")
        
        try:
            if obj is None:
                logger.debug("Serialized None value")
                return None
                
            # Handle lists
            if isinstance(obj, list):
                logger.debug(f"Serializing list with {len(obj)} items")
                return [CacheSerializer.serialize(item) for item in obj]
                
            # Handle dictionaries
            if isinstance(obj, dict):
                logger.debug(f"Serializing dict with {len(obj)} keys")
                return {key: CacheSerializer.serialize(value) for key, value in obj.items()}
            
            # Handle Pydantic models
            if hasattr(obj, 'model_dump'):
                logger.debug(f"Serializing Pydantic model: {obj.__class__.__name__}")
                return {
                    '__type__': 'pydantic_model',
                    '__class__': obj.__class__.__name__,
                    '__module__': obj.__class__.__module__,
                    '__data__': obj.model_dump()
                }
            
            # Handle SQLAlchemy models
            if hasattr(obj, '__table__'):
                logger.debug(f"Serializing SQLAlchemy model: {obj.__class__.__name__}")
                data = {}
                for column in obj.__table__.columns:
                    try:
                        value = getattr(obj, column.name)
                        if isinstance(value, (datetime, date)):
                            data[column.name] = value.isoformat()
                        elif isinstance(value, Decimal):
                            data[column.name] = float(value)
                        else:
                            data[column.name] = value
                    except Exception as e:
                        logger.warning(f"Failed to serialize column {column.name}: {str(e)}")
                        continue
                
                return {
                    '__type__': 'sqlalchemy_model',
                    '__class__': obj.__class__.__name__,
                    '__module__': obj.__class__.__module__,
                    '__data__': data
                }
            
            # Handle datetime objects
            if isinstance(obj, datetime):
                logger.debug("Serializing datetime object")
                return {
                    '__type__': 'datetime',
                    '__data__': obj.isoformat()
                }
            
            # Handle date objects
            if isinstance(obj, date):
                logger.debug("Serializing date object")
                return {
                    '__type__': 'date',
                    '__data__': obj.isoformat()
                }
            
            # Handle Decimal objects
            if isinstance(obj, Decimal):
                logger.debug("Serializing Decimal object")
                return {
                    '__type__': 'decimal',
                    '__data__': str(obj)
                }
            
            # For simple types, return as-is
            if isinstance(obj, (str, int, float, bool)):
                logger.debug(f"Serializing primitive type: {type(obj).__name__}")
                return obj
            
            # For complex objects, try to convert to dict
            try:
                if hasattr(obj, '__dict__'):
                    logger.debug(f"Serializing generic object: {obj.__class__.__name__}")
                    return {
                        '__type__': 'generic_object',
                        '__class__': obj.__class__.__name__,
                        '__module__': obj.__class__.__module__,
                        '__data__': CacheSerializer.serialize(obj.__dict__)
                    }
            except Exception as e:
                logger.warning(f"Failed to serialize object of type {type(obj)}: {e}")
            
            # Fallback: convert to string representation
            logger.debug(f"Using string representation for object of type {type(obj).__name__}")
            return {
                '__type__': 'string_repr',
                '__data__': str(obj)
            }
            
        except Exception as e:
            logger.error(f"Error during serialization: {str(e)}", exc_info=True)
            raise
        
        # This section is intentionally left blank as the logic has been moved into the try block
    
    @staticmethod
    def deserialize(obj: Any) -> Any:
        """
        Deserialize object from cache.
        
        Args:
            obj: The object to deserialize
            
        Returns:
            The deserialized object
        """
        logger.debug(f"Deserializing object of type: {type(obj).__name__ if obj is not None else 'None'}")
        
        try:
            if obj is None:
                logger.debug("Deserialized None value")
                return None
            
            # Handle lists
            if isinstance(obj, list):
                logger.debug(f"Deserializing list with {len(obj)} items")
                return [CacheSerializer.deserialize(item) for item in obj]
            
            # Handle dictionaries
            if isinstance(obj, dict):
                # Check if this is a special serialized object
                if '__type__' in obj:
                    obj_type = obj['__type__']
                    logger.debug(f"Deserializing special object of type: {obj_type}")
                    
                    if obj_type == 'datetime':
                        logger.debug("Converting to datetime")
                        return datetime.fromisoformat(obj['__data__'])
                    
                    elif obj_type == 'date':
                        logger.debug("Converting to date")
                        return date.fromisoformat(obj['__data__'])
                    
                    elif obj_type == 'decimal':
                        logger.debug("Converting to Decimal")
                        return Decimal(obj['__data__'])
                    
                    elif obj_type == 'string_repr':
                        logger.debug("Returning string representation")
                        return obj['__data__']
                    
                    elif obj_type in ('pydantic_model', 'sqlalchemy_model', 'generic_object'):
                        logger.debug(f"Deserializing {obj_type} data")
                        # For these types, just return the data dict
                        # The consuming code should handle the conversion
                        return CacheSerializer.deserialize(obj['__data__'])
                
                # Regular dictionary
                logger.debug(f"Deserializing dictionary with {len(obj)} keys")
                return {key: CacheSerializer.deserialize(value) for key, value in obj.items()}
            
            # For simple types, return as-is
            logger.debug(f"Returning as-is (type: {type(obj).__name__})")
            return obj
            
        except Exception as e:
            logger.error(f"Error during deserialization: {str(e)}", exc_info=True)
            # Return the original object if deserialization fails
            return obj


class CacheInvalidator:
    """Utility class for cache invalidation operations."""
    
    @staticmethod
    async def invalidate_tags(tags: List[str]) -> bool:
        """Invalidate all cache entries associated with given tags."""
        try:
            for tag in tags:
                tag_key = f"tag:{tag}"
                tagged_keys = await cache.get(tag_key)
                
                if tagged_keys and isinstance(tagged_keys, list):
                    # Delete all keys associated with this tag
                    for key in tagged_keys:
                        await cache.delete(key)
                        logger.debug(f"Invalidated cache key: {key}")
                    
                    # Clear the tag itself
                    await cache.delete(tag_key)
                    logger.info(f"Invalidated tag '{tag}' with {len(tagged_keys)} keys")
                else:
                    logger.debug(f"No keys found for tag: {tag}")
                    
            return True
        except Exception as e:
            logger.error(f"Error invalidating cache tags {tags}: {e}")
            return False


def cache_result(ttl: Optional[int] = None, 
                key_prefix: str = "",
                key_func: Optional[Callable] = None,
                tags: Optional[List[str]] = None,
                serialize: bool = True):
    """
    Enhanced caching decorator with tagging support and proper serialization.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        key_func: Custom key generation function
        tags: List of tags for cache invalidation
        serialize: Whether to use custom serialization (recommended for complex objects)
    """
    def decorator(func):
        func_name = f"{func.__module__}.{func.__name__}"
        logger.debug(f"Initializing cache_result decorator for {func_name} with ttl={ttl}, tags={tags}")
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = None
            try:
                # Generate cache key
                if key_func:
                    logger.debug(f"Using custom key function for {func_name}")
                    cache_key = key_func(*args, **kwargs)
                else:
                    logger.debug(f"Generating default cache key for {func_name}")
                    # Create a more robust key generation
                    sig = inspect.signature(func)
                    try:
                        bound_args = sig.bind(*args, **kwargs)
                        bound_args.apply_defaults()
                        # Filter out complex objects from key generation
                        simple_args = {}
                        for k, v in bound_args.arguments.items():
                            if isinstance(v, (str, int, float, bool, type(None))):
                                simple_args[k] = v
                            else:
                                simple_args[k] = str(type(v).__name__)
                        args_str = str(sorted(simple_args.items()))
                        logger.debug(f"Generated args string for {func_name}: {args_str}")
                    except Exception as e:
                        logger.warning(f"Error binding arguments for {func_name}: {e}")
                        # Fallback to simple string representation
                        args_str = f"args_len_{len(args)}_kwargs_len_{len(kwargs)}"
                        logger.debug(f"Using fallback args string: {args_str}")
                    
                    key_data = f"{func_name}:{args_str}"
                    cache_key = f"{key_prefix}{hashlib.md5(key_data.encode()).hexdigest()}"
                
                logger.debug(f"Generated cache key for {func_name}: {cache_key}")
                
                # Try cache first
                logger.debug(f"Checking cache for key: {cache_key}")
                cached_result = await cache.get(cache_key)
                
                if cached_result is not None:
                    logger.info(f"Cache HIT for {func_name} (key: {cache_key})")
                    if serialize:
                        logger.debug("Deserializing cached result")
                        return CacheSerializer.deserialize(cached_result)
                    logger.debug("Returning raw cached result")
                    return cached_result
                
                logger.debug(f"Cache MISS for {func_name} (key: {cache_key})")
                
                # Execute function
                logger.debug(f"Executing function: {func_name}")
                result = await func(*args, **kwargs)
                
                # Serialize result before caching if needed
                if serialize:
                    logger.debug("Serializing result for caching")
                    to_cache = CacheSerializer.serialize(result)
                else:
                    logger.debug("Using raw result for caching")
                    to_cache = result
                
                # Store in cache
                logger.debug(f"Caching result with TTL: {ttl} seconds")
                await cache.set(cache_key, to_cache, ttl=ttl)
                logger.info(f"Cached result for {func_name} (key: {cache_key})")
                
                # If tags are provided, associate them with the cache key
                if tags:
                    logger.debug(f"Associating tags with cache key: {tags}")
                    for tag in tags:
                        tag_key = f"tag:{tag}"
                        try:
                            # Get existing keys for this tag
                            tagged_keys = await cache.get(tag_key) or []
                            if not isinstance(tagged_keys, list):
                                logger.warning(f"Invalid tag data for {tag_key}, resetting to empty list")
                                tagged_keys = []
                            # Add current key if not already present
                            if cache_key not in tagged_keys:
                                logger.debug(f"Adding cache key {cache_key} to tag {tag}")
                                tagged_keys.append(cache_key)
                                await cache.set(tag_key, tagged_keys)
                                logger.debug(f"Updated tag {tag} with {len(tagged_keys)} keys")
                        except Exception as e:
                            logger.error(f"Error updating tag {tag}: {e}", exc_info=True)
                
                logger.debug(f"Successfully completed cache operation for {func_name}")
                return result
                
            except Exception as e:
                error_msg = f"Error in cache_result decorator for {func_name}"
                if cache_key:
                    error_msg += f" (key: {cache_key})"
                logger.error(f"{error_msg}: {e}", exc_info=True)
                # If there's an error with caching, still execute the function
                logger.debug("Falling back to direct function execution due to cache error")
                return await func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            logger.debug(f"Returning async wrapper for {func_name}")
            return async_wrapper
        else:
            # For synchronous functions, create a wrapper that runs in the event loop
            logger.debug(f"Creating sync wrapper for {func_name}")
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                logger.debug(f"Executing sync wrapper for {func_name}")
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're already in an event loop, create a new one
                        logger.debug("Creating new event loop for sync wrapper")
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    logger.debug("Running async wrapper in event loop")
                    return loop.run_until_complete(async_wrapper(*args, **kwargs))
                except Exception as e:
                    logger.error(f"Error in sync wrapper for {func_name}: {e}", exc_info=True)
                    # Fall back to direct function call
                    return func(*args, **kwargs)
            return sync_wrapper
        
        return sync_wrapper
    
    return decorator


def invalidate_cache(*tags: str):
    """
    Decorator to invalidate cache tags after function execution.
    
    Args:
        *tags: Variable length argument list of tag strings to invalidate
    """
    logger.debug(f"Initializing invalidate_cache decorator for tags: {tags}")
    
    def decorator(func):
        func_name = f"{func.__module__}.{func.__name__}"
        logger.debug(f"Setting up invalidate_cache for {func_name}")
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.debug(f"Executing async_wrapper for {func_name}")
            try:
                # Execute function first
                logger.debug(f"Calling function {func_name}")
                result = await func(*args, **kwargs)
                
                # Invalidate tags after successful execution
                if tags:
                    logger.info(f"Invalidating cache tags for {func_name}: {tags}")
                    try:
                        success = await CacheInvalidator.invalidate_tags(list(tags))
                        if success:
                            logger.info(f"Successfully invalidated cache for tags: {tags}")
                        else:
                            logger.warning(f"Failed to invalidate cache for tags: {tags}")
                    except Exception as e:
                        logger.error(f"Error during cache invalidation for {func_name}: {e}", 
                                    exc_info=True)
                else:
                    logger.debug(f"No tags provided for invalidation in {func_name}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in invalidate_cache decorator for {func_name}: {e}", 
                            exc_info=True)
                # Re-raise the original exception
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.debug(f"Executing sync_wrapper for {func_name}")
            try:
                # Execute function first
                logger.debug(f"Calling function {func_name}")
                result = func(*args, **kwargs)
                
                # Invalidate tags after successful execution
                if tags:
                    logger.info(f"Invalidating cache tags for {func_name}: {tags}")
                    try:
                        loop = asyncio.get_event_loop()
                        
                        if loop.is_running():
                            # Create a task for cache invalidation if we're in an event loop
                            logger.debug("Event loop is running, creating async task for invalidation")
                            asyncio.create_task(CacheInvalidator.invalidate_tags(list(tags)))
                            logger.info(f"Scheduled async cache invalidation for tags: {tags}")
                        else:
                            # Run synchronously in the current event loop
                            logger.debug("Running invalidation in current event loop")
                            loop.run_until_complete(CacheInvalidator.invalidate_tags(list(tags)))
                            logger.info(f"Synchronously invalidated cache for tags: {tags}")
                            
                    except RuntimeError as e:
                        # No event loop, create one
                        logger.debug("No event loop found, creating a new one")
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(CacheInvalidator.invalidate_tags(list(tags)))
                            logger.info(f"Invalidated cache in new event loop for tags: {tags}")
                        except Exception as inner_e:
                            logger.error(f"Failed to invalidate cache in new event loop: {inner_e}")
                            
                    except Exception as e:
                        logger.error(f"Unexpected error during cache invalidation: {e}", 
                                    exc_info=True)
                else:
                    logger.debug(f"No tags provided for invalidation in {func_name}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in sync invalidate_cache wrapper for {func_name}: {e}", 
                            exc_info=True)
                # Re-raise the original exception
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            logger.debug(f"Returning async wrapper for {func_name}")
            return async_wrapper
        else:
            logger.debug(f"Returning sync wrapper for {func_name}")
            return sync_wrapper
    
    return decorator


# Export all components
__all__ = [
    'cache_result', 
    'invalidate_cache', 
    'CacheInvalidator',
    'CacheSerializer'
]