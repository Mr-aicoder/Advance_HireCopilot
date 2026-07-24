import os
import sys
import asyncio
from dotenv import load_dotenv
import psycopg
from qdrant_client import AsyncQdrantClient
from neo4j import AsyncGraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

async def test_connections():
    print("--- Testing Cloud Connections ---\n")

    # 1. Test PostgreSQL (Neon)
    try:
        conn = await psycopg.AsyncConnection.connect(os.getenv("DATABASE_URL"))
        print("✅ PostgreSQL (Neon) Connected Successfully!")
        await conn.close()
    except Exception as e:
        print(f"❌ PostgreSQL Failed: {e}")

    # 2. Test Qdrant Cloud
    try:
        qdrant = AsyncQdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        await qdrant.get_collections()
        print("✅ Qdrant Cloud Connected Successfully!")
        await qdrant.close()
    except Exception as e:
        print(f"❌ Qdrant Failed: {e}")

    # 3. Test Neo4j AuraDB
    try:
        driver = AsyncGraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        await driver.verify_connectivity()
        print("✅ Neo4j AuraDB Connected Successfully!")
        await driver.close()
    except Exception as e:
        print(f"❌ Neo4j Failed: {e}")

    # 4. Test Gemini API
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        response = await llm.ainvoke("Ping test. Reply with 'Pong'.")
        print(f"✅ Gemini API Connected Successfully! Response: '{response.content.strip()}'")
    except Exception as e:
        print(f"❌ Gemini API Failed: {e}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(test_connections())