import psycopg
from psycopg_pool import AsyncConnectionPool
from config.settings import settings

# Global async connection pool for high-performance state check-pointing
_pool: AsyncConnectionPool | None = None

def get_postgres_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        if not settings.DATABASE_URL:
            raise ValueError("DATABASE_URL is not set in environment settings.")
        _pool = AsyncConnectionPool(conninfo=settings.DATABASE_URL, open=False)
    return _pool

async def get_postgres_connection():
    """Returns an async connection context manager to Neon Postgres."""
    if not settings.DATABASE_URL:
        raise ValueError("DATABASE_URL is not set in environment settings.")
    return await psycopg.AsyncConnection.connect(settings.DATABASE_URL)