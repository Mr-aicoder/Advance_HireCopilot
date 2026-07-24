from typing import TypedDict, List, Dict, Any, Optional

class HirerWorkflowState(TypedDict):
    candidate_id: str
    raw_resume_text: str
    target_role: str
    job_description: str
    key_requirements: List[str]
    
    # Outputs populated across nodes
    parsed_profile: Optional[Dict[str, Any]]
    screening_result: Optional[Dict[str, Any]]
    interview_plan: Optional[Dict[str, Any]]
    error: Optional[str]