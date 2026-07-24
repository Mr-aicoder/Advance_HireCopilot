import sys
import asyncio
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils.llm_provider import get_llm
from db.qdrant_client import get_async_qdrant_client
from db.neo4j_client import get_neo4j_driver
from db.postgres_client import get_postgres_connection

async def test_modules():
    print("--- Verifying Client Wrapper Modules ---\n")

    # LLM test
    llm = get_llm()
    print("✅ LLM Provider Initialized successfully!")

    # Qdrant client test
    q_client = get_async_qdrant_client()
    await q_client.get_collections()
    print("✅ Qdrant Client Wrapper functional!")
    await q_client.close()

    # Neo4j client test
    neo_driver = get_neo4j_driver()
    await neo_driver.verify_connectivity()
    print("✅ Neo4j Driver Wrapper functional!")
    await neo_driver.close()

    # Postgres client test
    async with await get_postgres_connection() as conn:
        print("✅ PostgreSQL Connection Wrapper functional!")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_modules())