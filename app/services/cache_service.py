import hashlib
import json
import redis.asyncio as aioredis
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

_pool: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    global _pool
    if _pool is None:
        _pool = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _pool


def _cache_key(query: str, depth: str) -> str:
    digest = hashlib.sha256(f"{query}:{depth}".encode()).hexdigest()[:16]
    return f"research:cache:{digest}"


async def get_cached_result(query: str, depth: str) -> dict | None:
    try:
        redis = _get_redis()
        key = _cache_key(query, depth)
        value = await redis.get(key)
        if value:
            logger.info("cache_hit", query=query[:50])
            return json.loads(value)
    except Exception as e:
        logger.warning("cache_get_failed", error=str(e))
    return None


async def set_cached_result(query: str, depth: str, result: dict) -> None:
    try:
        redis = _get_redis()
        key = _cache_key(query, depth)
        await redis.setex(key, settings.cache_ttl, json.dumps(result))
        logger.info("cache_set", query=query[:50], ttl=settings.cache_ttl)
    except Exception as e:
        logger.warning("cache_set_failed", error=str(e))
