"""
PDF parsing service.
Extracts text page-by-page and creates overlapping chunks with metadata.
"""
import fitz  # PyMuPDF
from dataclasses import dataclass
from config import get_settings

settings = get_settings()


@dataclass
class TextChunk:
    text: str
    page: int
    chunk_index: int
    document_id: str
    title: str


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """Return list of {page_number, text} dicts from a PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def chunk_pages(
    pages: list[dict],
    document_id: str,
    title: str,
) -> list[TextChunk]:
    """
    Split page text into overlapping chunks.
    Preserves which page each chunk came from.
    """
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    chunks: list[TextChunk] = []
    chunk_index = 0

    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]

        start = 0
        while start < len(text):
            end = start + size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        page=page_num,
                        chunk_index=chunk_index,
                        document_id=document_id,
                        title=title,
                    )
                )
                chunk_index += 1
            start += size - overlap  # slide window with overlap

    return chunks
