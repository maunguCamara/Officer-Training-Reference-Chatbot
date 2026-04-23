from pydantic import BaseModel
from typing import Optional


# ─── Auth ───────────────────────────────────────────────────────────────────

class TokenVerifyRequest(BaseModel):
    id_token: str


class UserInfo(BaseModel):
    uid: str
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None


# ─── Ingest ─────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    document_id: str
    title: str
    pages: int
    chunks_created: int
    message: str


class DocumentInfo(BaseModel):
    document_id: str
    title: str
    pages: int
    chunks: int
    uploaded_at: str


# ─── Chat ────────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    document_id: str
    title: str
    page: int
    excerpt: str          # short snippet from the source chunk


class ChatRequest(BaseModel):
    question: str
    language: str = "en"  # "en" for English, "sw" for Swahili
    conversation_history: list[dict] = []   # [{role, content}, ...]


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    language: str
