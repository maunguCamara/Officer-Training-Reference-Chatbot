"""
routers/ingest.py

Admin-only endpoint to upload PDFs, parse them, chunk them, and store embeddings.
Protected by a simple X-Admin-Secret header (no Firebase needed).
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Header
from models import IngestResponse, DocumentInfo
from services.pdf_parser import extract_pages, chunk_pages
from services.vector_store import store_chunks, list_documents, delete_document
from config import get_settings
import uuid
from datetime import datetime, timezone

settings = get_settings()
router = APIRouter(prefix="/ingest", tags=["ingest"])


def require_admin(x_admin_secret: str = Header(...)):
    if not settings.admin_secret or x_admin_secret != settings.admin_secret:
        raise HTTPException(status_code=403, detail="Invalid admin secret")


@router.post("", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    title: str = Form(...),
    x_admin_secret: str = Header(...),
):
    require_admin(x_admin_secret)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    document_id = str(uuid.uuid4())
    pages = extract_pages(pdf_bytes)
    if not pages:
        raise HTTPException(status_code=422, detail="No readable text found in PDF")

    chunks = chunk_pages(pages, document_id=document_id, title=title)
    if not chunks:
        raise HTTPException(status_code=422, detail="Could not create chunks from PDF")

    stored = store_chunks(chunks)

    return IngestResponse(
        document_id=document_id,
        title=title,
        pages=len(pages),
        chunks_created=stored,
        message=f"Successfully ingested '{title}' — {stored} chunks indexed.",
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def get_documents(x_admin_secret: str = Header(...)):
    require_admin(x_admin_secret)
    raw = list_documents()
    return [
        DocumentInfo(
            document_id=d["document_id"],
            title=d["title"],
            pages=d["max_page"],
            chunks=d["chunks"],
            uploaded_at=datetime.now(timezone.utc).isoformat(),
        )
        for d in raw
    ]


@router.delete("/documents/{document_id}")
async def remove_document(document_id: str, x_admin_secret: str = Header(...)):
    require_admin(x_admin_secret)
    deleted = delete_document(document_id)
    return {"document_id": document_id, "chunks_deleted": deleted}
