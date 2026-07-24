import sys
import asyncio
import json
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from agents.resume_parser_agent import parse_and_index_resume

sample_resume = """
Sachin Asaram Tupsamundar
Email: sachin@example.com
Location: Parbhani, India

Professional Summary:
AI Application Architect with over 3 years of experience in building enterprise-grade multi-agent AI systems, LangGraph orchestration, RAG pipelines, and cloud microservices.

Technical Skills:
- Python, FastAPI, Docker, Kubernetes
- LangChain, LangGraph, LangSmith
- PostgreSQL, Qdrant, Neo4j, Cypher
- AWS, GCP, CI/CD, Model Context Protocol (MCP)

Experience:
- Designed and deployed enterprise multi-agent workflows using LangGraph and Async PostgreSQL checkpointers.
- Built hybrid RAG architectures combining vector search in Qdrant with knowledge graph traversals in Neo4j.
"""

async def run_test():
    print("--- Running End-to-End Resume Parser Agent Test ---\n")
    result = await parse_and_index_resume(
        candidate_id="cand_001",
        raw_resume_text=sample_resume,
        target_role="AI Application Architect"
    )
    print("✅ Successfully Parsed & Dual-Indexed Resume:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_test())