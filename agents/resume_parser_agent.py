import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import get_llm
from rag.vector_rag import VectorRAGEngine
from rag.graph_rag import GraphRAGEngine

# Pydantic schema for Gemini structured output
class ParsedCandidateProfile(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address of candidate")
    skills: List[str] = Field(description="List of technical and soft skills extracted")
    experience_years: float = Field(description="Estimated total years of professional experience")
    summary_chunks: List[str] = Field(description="Key experience paragraphs or sections chunked for vector indexing")

async def parse_and_index_resume(candidate_id: str, raw_resume_text: str, target_role: str) -> Dict[str, Any]:
    """
    1. Uses Gemini 2.5 Flash to extract structured profile JSON from raw resume.
    2. Indexes summary chunks into Qdrant Vector Store.
    3. Builds Candidate -> HAS_SKILL & APPLIED_FOR relationships in Neo4j.
    """
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ParsedCandidateProfile)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical HR resume parser. Extract structured details accurately from the resume text."),
        ("human", "Resume Content:\n{resume_text}")
    ])

    chain = prompt | structured_llm
    parsed_data: ParsedCandidateProfile = await chain.ainvoke({"resume_text": raw_resume_text})

    # Initialize RAG Engines
    vector_rag = VectorRAGEngine()
    graph_rag = GraphRAGEngine()

    # 1. Vector RAG: Index resume text chunks into Qdrant
    if parsed_data.summary_chunks:
        await vector_rag.add_resume_chunks(
            candidate_id=candidate_id, 
            chunks=parsed_data.summary_chunks
        )

    # 2. GraphRAG: Construct Candidate -> HAS_SKILL & APPLIED_FOR nodes in Neo4j
    await graph_rag.add_candidate_graph(
        candidate_name=parsed_data.candidate_name,
        candidate_id=candidate_id,
        skills=parsed_data.skills,
        target_role=target_role
    )
    await graph_rag.close()

    return parsed_data.model_dump()