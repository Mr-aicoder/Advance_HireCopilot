from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from workflows.state import HirerWorkflowState
from agents.resume_parser_agent import parse_and_index_resume
from agents.screening_agent import screen_candidate
from agents.interview_question_agent import generate_interview_questions

# Node 1: Resume Parser & Dual-RAG Indexer
async def parse_and_index_node(state: HirerWorkflowState) -> Dict[str, Any]:
    try:
        profile = await parse_and_index_resume(
            candidate_id=state["candidate_id"],
            raw_resume_text=state["raw_resume_text"],
            target_role=state["target_role"]
        )
        return {"parsed_profile": profile}
    except Exception as e:
        return {"error": f"Resume Parsing Failed: {str(e)}"}

# Node 2: Candidate Screening Evaluator
async def screening_node(state: HirerWorkflowState) -> Dict[str, Any]:
    if state.get("error"):
        return {}
    try:
        screening = await screen_candidate(
            candidate_id=state["candidate_id"],
            job_description=state["job_description"],
            key_requirements=state["key_requirements"]
        )
        return {"screening_result": screening}
    except Exception as e:
        return {"error": f"Screening Failed: {str(e)}"}

# Node 3: Interview Question Generator
async def interview_gen_node(state: HirerWorkflowState) -> Dict[str, Any]:
    if state.get("error"):
        return {}
    try:
        verdict = state["screening_result"].get("screening_verdict", "Potential Match")
        interview = await generate_interview_questions(
            candidate_id=state["candidate_id"],
            target_role=state["target_role"],
            screening_verdict=verdict,
            key_requirements=state["key_requirements"]
        )
        return {"interview_plan": interview}
    except Exception as e:
        return {"error": f"Interview Gen Failed: {str(e)}"}

# Construct LangGraph StateGraph
def create_hirer_workflow():
    builder = StateGraph(HirerWorkflowState)

    # Add Nodes
    builder.add_node("parse_and_index", parse_and_index_node)
    builder.add_node("screen_candidate", screening_node)
    builder.add_node("generate_interview", interview_gen_node)

    # Define Control Flow Edges
    builder.add_edge(START, "parse_and_index")
    builder.add_edge("parse_and_index", "screen_candidate")
    builder.add_edge("screen_candidate", "generate_interview")
    builder.add_edge("generate_interview", END)

    return builder.compile()