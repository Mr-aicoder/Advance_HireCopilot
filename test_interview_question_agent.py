import sys
import asyncio
import json
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from agents.interview_question_agent import generate_interview_questions

async def run_test():
    print("--- Running End-to-End Interview Question Agent Test ---\n")
    result = await generate_interview_questions(
        candidate_id="cand_001",
        target_role="AI Application Architect",
        screening_verdict="Strong Match",
        key_requirements=["LangGraph checkpointers", "Qdrant and Neo4j hybrid RAG", "FastAPI microservices"]
    )
    print("✅ Interview Plan Generated Successfully:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_test())