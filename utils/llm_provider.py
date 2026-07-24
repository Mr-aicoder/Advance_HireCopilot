from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings

def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """
    Instantiates the Gemini 2.5 Flash LLM instance.
    High RPM/TPM throughput and massive context window for complex prompts.
    """
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing from environment settings.")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=temperature,
        max_retries=3,
    )