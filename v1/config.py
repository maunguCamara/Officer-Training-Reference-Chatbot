from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str

    # Twilio / WhatsApp
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = ""   # e.g. +14155238886
    whatsapp_verify_token: str = ""    # for Meta direct webhook verification

    # Admin (protects /ingest)
    admin_secret: str = ""

    # Vector Store
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "training_docs"

    # Embedding model (runs locally, no API cost)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Claude model
    claude_model: str = "claude-sonnet-4-20250514"
    claude_max_tokens: int = 1500

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # RAG retrieval
    top_k_chunks: int = 5

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
