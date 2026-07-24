from qdrant_client import AsyncQdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import settings

COLLECTION_NAME = "resume_chunks"

def get_async_qdrant_client() -> AsyncQdrantClient:
    """Instantiates the async native Qdrant client."""
    if not settings.QDRANT_URL or not settings.QDRANT_API_KEY:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY is missing from settings.")
    return AsyncQdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY
    )

def get_qdrant_vector_store() -> QdrantVectorStore:
    """Instantiates LangChain's Qdrant Vector Store wrapper with Gemini Embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.GOOGLE_API_KEY
    )
    
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY
    )