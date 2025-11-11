import os
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@lru_cache
def get_settings():
    return {
        "REDIS_URL": os.getenv("REDIS_URL"),
        "RATE_LIMIT": os.getenv("RATE_LIMIT", "60/minute"),
        "CACHE_TTL_SECONDS": int(os.getenv("CACHE_TTL_SECONDS", "600")),
    }

def get_redis() -> Optional[object]:
    url = get_settings().get("REDIS_URL")
    if not url:
        return None
    try:
        from redis import asyncio as aioredis
        return aioredis.from_url(url, encoding="utf-8", decode_responses=True)
    except Exception:
        return None
