import sys
import asyncio
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from rag.vector_rag import VectorRAGEngine
from rag.graph_rag import GraphRAGEngine
from db.qdrant_client import get_async_qdrant_client, COLLECTION_NAME

async def test_rag_engines():
    print("--- Testing Dual-RAG Engines ---\n")

    # Clean up old Qdrant test collection if exists
    qdrant_client = get_async_qdrant_client()
    collections = await qdrant_client.get_collections()
    if COLLECTION_NAME in [c.name for c in collections.collections]:
        await qdrant_client.delete_collection(COLLECTION_NAME)
    await qdrant_client.close()

    vector_rag = VectorRAGEngine()
    graph_rag = GraphRAGEngine()

    candidate_id = "cand_test_001"
    candidate_name = "Sachin Tupsamundar"
    sample_chunks = [
        "Experienced in orchestrating multi-agent systems using LangGraph and LangChain.",
        "Built RAG pipelines with vector databases like Qdrant and knowledge graphs like Neo4j.",
        "Proficient in Python, FastAPI, Docker, PostgreSQL, and cloud deployments."
    ]
    skills = ["python", "fastapi", "langgraph", "langchain", "qdrant", "neo4j", "docker"]
    role = "AI Application Architect"

    # 1. Test Vector RAG
    print("1. Indexing resume chunks into Qdrant...")
    doc_ids = await vector_rag.add_resume_chunks(candidate_id, sample_chunks)
    print(f"   ✅ Indexed {len(doc_ids)} chunks into Qdrant.")

    print("2. Performing similarity search in Qdrant...")
    search_results = await vector_rag.search_relevant_chunks("What vector databases and graph tools does the candidate know?")
    for idx, doc in enumerate(search_results, 1):
        print(f"   Result {idx}: {doc.page_content}")

    # 2. Test Graph RAG
    print("\n3. Building Candidate graph entities in Neo4j...")
    await graph_rag.add_candidate_graph(
        candidate_name=candidate_name,
        candidate_id=candidate_id,
        skills=skills,
        target_role=role
    )
    print("   ✅ Graph relationships created in Neo4j.")

    print("4. Querying candidate skills from Neo4j...")
    graph_data = await graph_rag.get_candidate_skills(candidate_id)
    print(f"   ✅ Found Candidate: {graph_data['name']}")
    print(f"   ✅ Skills in Graph: {graph_data['skills']}")

    await graph_rag.close()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_rag_engines())