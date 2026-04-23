"""
ChromaDB vector store interface.
Handles embedding, storing, and retrieving document chunks.
"""
import chromadb
from chromadb.utils import embedding_functions
from services.pdf_parser import TextChunk
from config import get_settings
from datetime import datetime, timezone

settings = get_settings()

# Singleton client
_client: chromadb.PersistentClient | None = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def store_chunks(chunks: list[TextChunk]) -> int:
    """Embed and store chunks. Returns number stored."""
    col = _get_collection()

    ids = [f"{c.document_id}__chunk_{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "document_id": c.document_id,
            "title": c.title,
            "page": c.page,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    # ChromaDB handles embedding via the collection's embedding_function
    col.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def query_chunks(question: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve the most relevant chunks for a question.
    Returns list of {text, document_id, title, page, score} dicts.
    """
    col = _get_collection()
    results = col.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "text": doc,
                    "document_id": meta["document_id"],
                    "title": meta["title"],
                    "page": meta["page"],
                    "score": round(1 - dist, 4),  # cosine similarity
                }
            )
    return chunks


def list_documents() -> list[dict]:
    """Return unique documents stored in the vector DB."""
    col = _get_collection()
    # Get all metadata
    result = col.get(include=["metadatas"])
    seen = {}
    for meta in result["metadatas"]:
        doc_id = meta["document_id"]
        if doc_id not in seen:
            seen[doc_id] = {
                "document_id": doc_id,
                "title": meta["title"],
                "chunks": 0,
                "max_page": 0,
            }
        seen[doc_id]["chunks"] += 1
        seen[doc_id]["max_page"] = max(seen[doc_id]["max_page"], meta["page"])
    return list(seen.values())


def delete_document(document_id: str) -> int:
    """Delete all chunks for a document. Returns count deleted."""
    col = _get_collection()
    existing = col.get(where={"document_id": document_id})
    if existing["ids"]:
        col.delete(ids=existing["ids"])
    return len(existing["ids"])
