from typing import List, Dict, Any
from db.neo4j_client import get_neo4j_driver

class GraphRAGEngine:
    def __init__(self):
        self.driver = get_neo4j_driver()

    async def add_candidate_graph(self, candidate_name: str, candidate_id: str, skills: List[str], target_role: str):
        """Constructs Candidate -> HAS_SKILL & Candidate -> APPLIED_FOR relationships in Neo4j."""
        query = """
        MERGE (c:Candidate {id: $candidate_id})
        SET c.name = $candidate_name
        
        MERGE (r:Role {title: $target_role})
        MERGE (c)-[:APPLIED_FOR]->(r)
        
        WITH c
        UNWIND $skills AS skill_name
        MERGE (s:Skill {name: toLower(trim(skill_name))})
        MERGE (c)-[:HAS_SKILL]->(s)
        """
        # Remove database="neo4j" so Aura uses its home database routing automatically
        async with self.driver.session() as session:
            await session.run(
                query,
                candidate_id=candidate_id,
                candidate_name=candidate_name,
                skills=skills,
                target_role=target_role
            )

    async def get_candidate_skills(self, candidate_id: str) -> Dict[str, Any]:
        """Queries entity nodes from Neo4j graph using Cypher."""
        query = """
        MATCH (c:Candidate {id: $candidate_id})-[:HAS_SKILL]->(s:Skill)
        RETURN c.name AS name, collect(s.name) AS skills
        """
        async with self.driver.session() as session:
            result = await session.run(query, candidate_id=candidate_id)
            record = await result.single()
            if record:
                return {"name": record["name"], "skills": record["skills"]}
            return {"name": "", "skills": []}

    async def close(self):
        """Closes the driver connection."""
        await self.driver.close()