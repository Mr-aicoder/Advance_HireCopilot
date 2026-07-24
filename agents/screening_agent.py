from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import get_llm
from rag.vector_rag import VectorRAGEngine
from rag.graph_rag import GraphRAGEngine

# Pydantic schema for evaluation output
class ScreeningResult(BaseModel):
    match_score: int = Field(description="Match score between 0 and 100 based on job requirements")
    strengths: List[str] = Field(description="Key strengths matching the position requirements")
    missing_skills: List[str] = Field(description="Key skills or experience gaps identified")
    screening_verdict: str = Field(description="Verdict: Strong Match, Potential Match, or Not Recommended")
    reasoning: str = Field(description="Detailed explanation justifying the score and verdict")

async def screen_candidate(candidate_id: str, job_description: str, key_requirements: List[str]) -> Dict[str, Any]:
    """
    Evaluates a candidate against a job description using hybrid Dual-RAG context.
    """
    vector_rag = VectorRAGEngine()
    graph_rag = GraphRAGEngine()

    # 1. Fetch skills & entities from Neo4j Graph
    graph_data = await graph_rag.get_candidate_skills(candidate_id)
    candidate_skills = graph_data.get("skills", [])
    candidate_name = graph_data.get("name", "Unknown Candidate")

    # 2. Query Qdrant for semantic context matching the job requirements
    requirement_query = f"Experience and projects relevant to: {' '.join(key_requirements)}"
    relevant_docs = await vector_rag.search_relevant_chunks(query=requirement_query, limit=3)
    retrieved_context = "\n".join([doc.page_content for doc in relevant_docs])

    # 3. LLM Structured Evaluation using Gemini
    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(ScreeningResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert technical recruiter evaluating candidate fit.
Use both the verified graph skills and retrieved experience excerpts to produce a objective assessment."""),
        ("human", """Candidate Name: {candidate_name}
Target Job Description:
{job_description}

Key Skills in Candidate Graph (Neo4j):
{candidate_skills}

Relevant Resume Content Excerpts (Qdrant):
{retrieved_context}

Provide a structured candidate screening evaluation.""")
    ])

    chain = prompt | structured_llm
    evaluation: ScreeningResult = await chain.ainvoke({
        "candidate_name": candidate_name,
        "job_description": job_description,
        "candidate_skills": ", ".join(candidate_skills),
        "retrieved_context": retrieved_context
    })

    await graph_rag.close()
    return evaluation.model_dump()