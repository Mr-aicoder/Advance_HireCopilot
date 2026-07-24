import sys
import asyncio
import json
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from agents.screening_agent import screen_candidate

job_desc = """
We are looking for a Senior AI Application Architect to build agentic workflows, multi-agent orchestration engines, 
and enterprise RAG pipelines. 
Key requirements: Python, LangGraph, Qdrant/Neo4j, Docker, and PostgreSQL.
"""

key_reqs = ["LangGraph multi-agent systems", "Hybrid RAG with Qdrant and Neo4j", "FastAPI and PostgreSQL"]

async def run_test():
    print("--- Running End-to-End Candidate Screening Agent Test ---\n")
    # Candidate 'cand_001' was indexed during the resume parser test
    result = await screen_candidate(
        candidate_id="cand_001",
        job_description=job_desc,
        key_requirements=key_reqs
    )
    print("✅ Screening Evaluation Completed:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_test())