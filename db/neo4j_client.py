from neo4j import AsyncGraphDatabase
from config.settings import settings

def get_neo4j_driver():
    """Returns an Async Neo4j Driver instance for Graph queries."""
    if not settings.NEO4J_URI or not settings.NEO4J_PASSWORD:
        raise ValueError("NEO4J_URI or NEO4J_PASSWORD is missing from settings.")
    
    return AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
    )