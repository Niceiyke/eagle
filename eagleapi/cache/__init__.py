# cache/__init__.py
"""
Eagle Cache Framework - Flexible caching system for Eagle API
Supports in-memory, Redis, and Memcached backends with a unified interface.
"""

from .core import Cache, cache_manager
from.config import setup_eagle_cache, CacheConfig
from .decorators import cached, cache_key, invalidate_cache
from .backends import InMemoryBackend, RedisBackend, MemcachedBackend
from .serializers import JSONSerializer, PickleSerializer, CompressedSerializer

__all__ = [
    'Cache',
    'cache_manager', 
    'cached',
    'cache_key',
    'invalidate_cache',
    'InMemoryBackend',
    'RedisBackend', 
    'MemcachedBackend',
    'JSONSerializer',
    'PickleSerializer',
    'CompressedSerializer',
    'setup_eagle_cache',
    'CacheConfig'
]
