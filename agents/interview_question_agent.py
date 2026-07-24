from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import get_llm
from rag.vector_rag import VectorRAGEngine
from rag.graph_rag import GraphRAGEngine

class InterviewQuestion(BaseModel):
    category: str = Field(description="Category e.g., 'System Architecture', 'LangGraph Orchestration', 'Database Design'")
    question: str = Field(description="The technical or behavioral question to ask")
    expected_signal: str = Field(description="What a strong answer should demonstrate")
    difficulty: str = Field(description="Difficulty level: Easy, Medium, Hard, or Deep Dive")

class InterviewPlan(BaseModel):
    candidate_name: str = Field(description="Full name of candidate")
    target_role: str = Field(description="Role being interviewed for")
    questions: List[InterviewQuestion] = Field(description="List of tailored interview questions")

async def generate_interview_questions(
    candidate_id: str, 
    target_role: str, 
    screening_verdict: str,
    key_requirements: List[str]
) -> Dict[str, Any]:
    """
    Generates targeted technical interview questions based on candidate graph skills 
    and vector experience chunks.
    """
    vector_rag = VectorRAGEngine()
    graph_rag = GraphRAGEngine()

    # 1. Retrieve Candidate Skills from Neo4j
    graph_data = await graph_rag.get_candidate_skills(candidate_id)
    candidate_skills = graph_data.get("skills", [])
    candidate_name = graph_data.get("name", "Candidate")

    # 2. Retrieve Detailed Experience Chunks from Qdrant
    req_query = " ".join(key_requirements)
    relevant_docs = await vector_rag.search_relevant_chunks(query=req_query, limit=3)
    retrieved_context = "\n".join([doc.page_content for doc in relevant_docs])

    # 3. LLM Question Generation
    llm = get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(InterviewPlan)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Principal AI Architect leading technical interviews.
Generate sharp, non-generic interview questions tailored specifically to the candidate's actual projects, 
skills graph, and target role requirements."""),
        ("human", """Candidate Name: {candidate_name}
Target Role: {target_role}
Screening Verdict: {screening_verdict}

Verified Skills (Neo4j):
{candidate_skills}

Resume Context Excerpts (Qdrant):
{retrieved_context}

Generate 4-5 tailored technical and architectural interview questions with expected signals.""")
    ])

    chain = prompt | structured_llm
    plan: InterviewPlan = await chain.ainvoke({
        "candidate_name": candidate_name,
        "target_role": target_role,
        "screening_verdict": screening_verdict,
        "candidate_skills": ", ".join(candidate_skills),
        "retrieved_context": retrieved_context
    })

    await graph_rag.close()
    return plan.model_dump()