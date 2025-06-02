# cache/config.py
"""Cache configuration and setup utilities."""

import os
from typing import Optional, Dict, Any
from fastapi import HTTPException
from .core import Cache, cache_manager
from .backends import InMemoryBackend, RedisBackend, MemcachedBackend
from .serializers import JSONSerializer, PickleSerializer, CompressedSerializer


class CacheConfig:
    """Cache configuration helper."""
    
    @staticmethod
    def from_url(
        url: str, 
        name: str = "default",
        serializer: str = "json",
        **kwargs
    ) -> Cache:
        """
        Create cache from URL.
        
        Supported URLs:
        - memory://
        - redis://localhost:6379/0
        - memcached://localhost:11211
        
        Args:
            url: Cache backend URL
            name: Cache instance name
            serializer: Serializer type (json, pickle, compressed)
            **kwargs: Additional configuration
        """
        # Parse URL
        if url.startswith("memory://"):
            backend = InMemoryBackend(**kwargs)
        elif url.startswith("redis://"):
            backend = RedisBackend(url=url, **kwargs)
        elif url.startswith("memcached://"):
            # Extract servers from URL
            servers = [url.replace("memcached://", "")]
            backend = MemcachedBackend(servers=servers, **kwargs)
        else:
            raise ValueError(f"Unsupported cache URL: {url}")
        
        # Create serializer
        if serializer == "json":
            serializer_instance = JSONSerializer()
        elif serializer == "pickle":
            serializer_instance = PickleSerializer()
        elif serializer == "compressed":
            base_serializer = kwargs.get("base_serializer", "json")
            if base_serializer == "json":
                base = JSONSerializer()
            else:
                base = PickleSerializer()
            serializer_instance = CompressedSerializer(base)
        else:
            serializer_instance = None
        
        # Create cache
        cache = Cache(
            backend=backend,
            serializer=serializer_instance,
            **{k: v for k, v in kwargs.items() 
               if k not in ['base_serializer']}
        )
        
        # Register cache
        cache_manager.register_cache(name, cache, is_default=name == "default")
        
        return cache
    
    @staticmethod
    def from_env(prefix: str = "EAGLE_CACHE") -> Optional[Cache]:
        """
        Create cache from environment variables.
        
        Environment variables:
        - EAGLE_CACHE_URL: Cache backend URL
        - EAGLE_CACHE_TTL: Default TTL
        - EAGLE_CACHE_SERIALIZER: Serializer type
        - EAGLE_CACHE_KEY_PREFIX: Key prefix
        """
        url = os.getenv(f"{prefix}_URL")
        if not url:
            return None
        
        config = {
            "default_ttl": int(os.getenv(f"{prefix}_TTL", "3600")),
            "key_prefix": os.getenv(f"{prefix}_KEY_PREFIX", "eagle:"),
            "serializer": os.getenv(f"{prefix}_SERIALIZER", "json"),
        }
        
        return CacheConfig.from_url(url, **config)
    
    @staticmethod
    def setup_default_cache(
        backend: str = "memory",
        redis_url: str = "redis://localhost:6379/0",
        **kwargs
    ) -> Cache:
        """
        Set up default cache with simple configuration.
        
        Args:
            backend: Backend type ("memory", "redis", "memcached")
            redis_url: Redis URL if using Redis backend
            **kwargs: Additional configuration
        """
        if backend == "memory":
            url = "memory://"
        elif backend == "redis":
            url = redis_url
        elif backend == "memcached":
            url = "memcached://localhost:11211"
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        return CacheConfig.from_url(url, **kwargs)

# Integration with Eagle API
def setup_eagle_cache(app, cache_config: Optional[Dict[str, Any]] = None):
    """
    Set up caching for Eagle API application.
    
    Args:
        app: Eagle API application instance
        cache_config: Cache configuration dictionary
    """
    if cache_config is None:
        # Try to load from environment
        cache = CacheConfig.from_env()
        if cache is None:
            # Default to in-memory cache
            cache = CacheConfig.setup_default_cache()
    else:
        # Create cache from config
        backend_type = cache_config.get("backend", "memory")
        if backend_type == "redis":
            url = cache_config.get("redis_url", "redis://localhost:6379/0")
        elif backend_type == "memcached":
            servers = cache_config.get("servers", ["localhost:11211"])
            url = f"memcached://{servers[0]}"
        else:
            url = "memory://"
        
        cache = CacheConfig.from_url(
            url,
            serializer=cache_config.get("serializer", "json"),
            default_ttl=cache_config.get("default_ttl", 3600),
            key_prefix=cache_config.get("key_prefix", "eagle:")
        )
    
    # Add cache cleanup on shutdown
    @app.on_event("shutdown")
    async def cleanup_cache():
        await cache_manager.close_all()
    
    # Add cache stats endpoint
    @app.get("/cache/stats", include_in_schema=False)
    async def cache_stats():
        """Get cache statistics."""
        stats = {}
        for name in cache_manager.list_caches():
            cache = cache_manager.get_cache(name)
            if cache:
                cache_stats = await cache.stats()
                stats[name] = {
                    "hits": cache_stats.hits,
                    "misses": cache_stats.misses,
                    "hit_rate": cache_stats.hit_rate,
                    "sets": cache_stats.sets,
                    "deletes": cache_stats.deletes,
                    "errors": cache_stats.errors,
                    "total_time": cache_stats.total_time
                }
        return stats
    
    # Add cache management endpoints
    @app.post("/cache/clear", include_in_schema=False)
    async def clear_cache(cache_name: Optional[str] = None):
        """Clear cache entries."""
        cache = cache_manager.get_cache(cache_name)
        if not cache:
            raise HTTPException(status_code=404, detail="Cache not found")
        
        success = await cache.clear()
        return {"success": success}
    
    @app.delete("/cache/keys/{key}", include_in_schema=False)
    async def delete_cache_key(key: str, cache_name: Optional[str] = None):
        """Delete specific cache key."""
        cache = cache_manager.get_cache(cache_name)
        if not cache:
            raise HTTPException(status_code=404, detail="Cache not found")
        
        success = await cache.delete(key)
        return {"success": success}
    
    app.logger.info("Cache system initialized")
    return cache