# Usage Examples for Eagle Cache Framework

# 1. Basic Setup in your Eagle API application
from eagleapi import create_app
from eagleapi.cache import setup_eagle_cache, CacheConfig

# Method 1: Auto-setup from environment variables
app = create_app()
# Set environment variables:
# EAGLE_CACHE_URL=redis://localhost:6379/0
# EAGLE_CACHE_TTL=3600
# EAGLE_CACHE_SERIALIZER=json
setup_eagle_cache(app)

# Method 2: Manual configuration
app = create_app()
cache_config = {
    "backend": "redis",
    "redis_url": "redis://localhost:6379/0",
    "serializer": "json",
    "default_ttl": 3600,
    "key_prefix": "myapp:"
}
setup_eagle_cache(app, cache_config)

# Method 3: Simple in-memory cache (default)
app = create_app()
setup_eagle_cache(app)  # Uses in-memory cache by default

# 2. Using Cache Decorators
from eagleapi.cache import cached, invalidate_cache, cache_key

# Basic caching with auto-generated key
@cached(ttl=300)  # Cache for 5 minutes
async def get_user_data(user_id: int):
    # Expensive database operation
    return await db.query("SELECT * FROM users WHERE id = ?", user_id)

# Custom cache key
@cached(ttl=600, key="user_profile:{0}")  # {0} refers to first argument
async def get_user_profile(user_id: int):
    return await db.get_user_profile(user_id)

# Cache with conditional skipping
@cached(
    ttl=300,
    skip_cache=lambda user_id, force_refresh=False: force_refresh
)
async def get_user_posts(user_id: int, force_refresh: bool = False):
    return await db.get_user_posts(user_id)

# Cache invalidation
@invalidate_cache(pattern="user:*")
async def update_user(user_id: int, data: dict):
    result = await db.update_user(user_id, data)
    return result

# 3. Manual Cache Operations
from eagleapi.cache import cache_manager

async def manual_cache_example():
    cache = cache_manager.get_cache()  # Get default cache
    
    # Basic operations
    await cache.set("key1", "value1", ttl=300)
    value = await cache.get("key1")
    exists = await cache.exists("key1")
    await cache.delete("key1")
    
    # Batch operations
    await cache.set_many({
        "key1": "value1",
        "key2": "value2",
        "key3": "value3"
    }, ttl=600)
    
    values = await cache.get_many(["key1", "key2", "key3"])
    print(values)  # {"key1": "value1", "key2": "value2", "key3": "value3"}
    
    # Get or set pattern
    user_data = await cache.get_or_set(
        "user:123",
        factory=lambda: get_user_from_db(123),
        ttl=300
    )
    
    # Pattern-based invalidation
    await cache.invalidate_pattern("user:*")

# 4. FastAPI Route Caching
from fastapi import APIRouter, Depends
from eagleapi.cache import cached

router = APIRouter()

# Cache API responses
@router.get("/users/{user_id}")
@cached(ttl=300, key="api:user:{user_id}")
async def get_user_endpoint(user_id: int):
    user = await get_user_from_db(user_id)
    return user

# Cache with query parameters
@router.get("/posts")
@cached(ttl=120)  # Auto-generates key from all parameters
async def get_posts(page: int = 1, limit: int = 10, category: str = None):
    posts = await get_posts_from_db(page, limit, category)
    return posts

# 5. Advanced Configuration Examples

# Multiple cache instances
from eagleapi.cache import CacheConfig, cache_manager

# Session cache (Redis)
session_cache = CacheConfig.from_url(
    "redis://localhost:6379/1",
    name="sessions",
    serializer="pickle",
    default_ttl=1800,  # 30 minutes
    key_prefix="session:"
)

# API response cache (In-memory)
api_cache = CacheConfig.from_url(
    "memory://",
    name="api_responses",
    serializer="json",
    default_ttl=300,  # 5 minutes
    key_prefix="api:"
)

# Use specific cache instance
@cached(ttl=1800, cache_name="sessions")
async def get_user_session(session_id: str):
    return await db.get_session(session_id)

# 6. Cache with Compression
compressed_cache = CacheConfig.from_url(
    "redis://localhost:6379/2",
    name="large_data",
    serializer="compressed",
    base_serializer="pickle",
    default_ttl=3600
)

@cached(ttl=3600, cache_name="large_data")
async def get_large_dataset():
    # This will be compressed before storing
    return await db.get_large_data()

# 7. Cache Statistics and Monitoring
async def cache_monitoring():
    cache = cache_manager.get_cache()
    stats = await cache.stats()
    
    print(f"Cache Hit Rate: {stats.hit_rate:.2f}%")
    print(f"Total Hits: {stats.hits}")
    print(f"Total Misses: {stats.misses}")
    print(f"Total Sets: {stats.sets}")
    print(f"Average Response Time: {stats.total_time / (stats.hits + stats.misses):.4f}s")

# 8. Custom Serialization
from eagleapi.cache.serializers import Serializer
import msgpack

class MsgPackSerializer(Serializer):
    def serialize(self, data):
        return msgpack.packb(data)
    
    def deserialize(self, data):
        return msgpack.unpackb(data, raw=False)

# Use custom serializer
custom_cache = CacheConfig.from_url(
    "redis://localhost:6379/3",
    name="msgpack_cache"
)
custom_cache.serializer = MsgPackSerializer()

# 9. Environment-based Configuration
# .env file:
# EAGLE_CACHE_URL=redis://localhost:6379/0
# EAGLE_CACHE_TTL=3600
# EAGLE_CACHE_SERIALIZER=json
# EAGLE_CACHE_KEY_PREFIX=myapp:

import os
from dotenv import load_dotenv

load_dotenv()

app = create_app()
cache = CacheConfig.from_env()  # Automatically configured from .env

# 10. Production Setup Example
def setup_production_cache(app):
    """Production-ready cache setup with Redis cluster."""
    
    # Primary cache (Redis cluster)
    primary_cache = CacheConfig.from_url(
        os.getenv("REDIS_CLUSTER_URL", "redis://localhost:6379/0"),
        name="primary",
        serializer="compressed",
        base_serializer="pickle",
        default_ttl=3600,
        key_prefix=f"{os.getenv('APP_NAME', 'eagle')}:"
    )
    
    # Session cache (separate Redis instance)
    session_cache = CacheConfig.from_url(
        os.getenv("REDIS_SESSION_URL", "redis://localhost:6379/1"),
        name="sessions",
        serializer="pickle",
        default_ttl=1800,
        key_prefix="sess:"
    )
    
    # API response cache (in-memory for fast access)
    api_cache = CacheConfig.from_url(
        "memory://",
        name="api_responses",
        max_size=10000,
        cleanup_interval=30,
        default_ttl=300,
        key_prefix="api:"
    )
    
    setup_eagle_cache(app)
    
    # Health check for cache
    @app.get("/health/cache")
    async def cache_health():
        try:
            cache = cache_manager.get_cache("primary")
            await cache.set("health_check", "ok", ttl=10)
            result = await cache.get("health_check")
            await cache.delete("health_check")
            
            return {
                "status": "healthy" if result == "ok" else "unhealthy",
                "caches": cache_manager.list_caches()
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

# 11. Cache Warming Strategy
async def warm_cache():
    """Pre-populate cache with frequently accessed data."""
    cache = cache_manager.get_cache()
    
    # Warm up user data
    popular_users = await db.get_popular_users(limit=100)
    for user in popular_users:
        await cache.set(f"user:{user.id}", user, ttl=3600)
    
    # Warm up configuration
    config = await db.get_app_config()
    await cache.set("app_config", config, ttl=7200)
    
    print(f"Cache warmed with {len(popular_users)} users and config")

# Add to startup
@app.on_event("startup")
async def startup():
    await warm_cache()

# 12. Error Handling and Fallbacks
from eagleapi.cache import cached

@cached(ttl=300)
async def resilient_function(param1: str):
    try:
        # If cache fails, function still executes
        result = await expensive_operation(param1)
        return result
    except Exception as e:
        logger.error(f"Error in cached function: {e}")
        # Function can implement its own fallback logic
        return await fallback_operation(param1)