import sys
import asyncio
import json
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from workflows.hirer_workflow import create_hirer_workflow

sample_resume = """
Sachin Asaram Tupsamundar
Email: sachin@example.com

Summary:
AI Application Architect with 3+ years experience building multi-agent systems using LangGraph, RAG pipelines, FastAPI, PostgreSQL, and Qdrant/Neo4j.
"""

job_desc = """
Looking for a Senior AI Application Architect to lead agentic workflow designs, dual-RAG architectures, and microservice integration.
"""

async def run_workflow_test():
    print("--- Executing Full LangGraph Multi-Agent Workflow ---\n")
    
    workflow = create_hirer_workflow()
    
    initial_state = {
        "candidate_id": "cand_wf_101",
        "raw_resume_text": sample_resume,
        "target_role": "AI Application Architect",
        "job_description": job_desc,
        "key_requirements": ["LangGraph agentic workflows", "Qdrant and Neo4j Dual-RAG", "FastAPI microservices"],
        "parsed_profile": None,
        "screening_result": None,
        "interview_plan": None,
        "error": None
    }

    final_state = await workflow.ainvoke(initial_state)

    if final_state.get("error"):
        print(f"❌ Workflow Error: {final_state['error']}")
    else:
        print("✅ Workflow Completed Successfully!\n")
        print("--- Final State Highlights ---")
        print(f"Candidate: {final_state['parsed_profile']['candidate_name']}")
        print(f"Match Score: {final_state['screening_result']['match_score']}%")
        print(f"Verdict: {final_state['screening_result']['screening_verdict']}")
        print(f"Interview Questions Generated: {len(final_state['interview_plan']['questions'])}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_workflow_test())