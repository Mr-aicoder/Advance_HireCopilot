from typing import List, Dict, Any
from langchain_core.documents import Document
from db.qdrant_client import get_async_qdrant_client, get_qdrant_vector_store, COLLECTION_NAME
from qdrant_client.models import VectorParams, Distance

class VectorRAGEngine:
    def __init__(self):
        self.qdrant_client = get_async_qdrant_client()

    async def ensure_collection_exists(self):
        """Ensures the Qdrant vector collection exists before adding documents."""
        collections = await self.qdrant_client.get_collections()
        existing_names = [c.name for c in collections.collections]
        
        # If collection exists, verify/recreate if needed; otherwise create with 3072 dims
        if COLLECTION_NAME not in existing_names:
            await self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
            )

    async def add_resume_chunks(self, candidate_id: str, chunks: List[str]) -> List[str]:
        """Indexes candidate resume chunks into Qdrant vector database."""
        await self.ensure_collection_exists()
        
        vector_store = get_qdrant_vector_store()
        documents = [
            Document(
                page_content=chunk,
                metadata={"candidate_id": candidate_id}
            )
            for chunk in chunks
        ]
        
        ids = await vector_store.aadd_documents(documents)
        return ids

    async def search_relevant_chunks(self, query: str, limit: int = 4) -> List[Document]:
        """Performs semantic similarity search across stored resume chunks."""
        vector_store = get_qdrant_vector_store()
        results = await vector_store.asimilarity_search(query, k=limit)
        return results