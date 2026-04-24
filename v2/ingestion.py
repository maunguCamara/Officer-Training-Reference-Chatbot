# ingestion.py
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import fitz
from langdetect import detect, DetectorFactory
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

DetectorFactory.seed = 0

PDF_DIR = Path("data/pdfs")
CHROMA_DB_DIR = Path("data/chroma_db")


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

if LLM_PROVIDER == "ollama":
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
else:
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def extract_text_from_pdfs(pdf_dir=PDF_DIR):
    all_docs = []
    for pdf_file in pdf_dir.glob("*.pdf"):
        doc = fitz.open(pdf_file)
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text").strip()
            if not text:
                continue
            text = " ".join(text.split())
            metadata = {
                "source": pdf_file.name,
                "page": page_num + 1
            }
            all_docs.append(Document(page_content=text, metadata=metadata))
        doc.close()
    return all_docs

def split_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def add_language_metadata(chunks):
    for chunk in chunks:
        try:
            lang = detect(chunk.page_content)
            if lang.startswith("en"): chunk.metadata["language"] = "en"
            elif lang.startswith("sw"): chunk.metadata["language"] = "sw"
            else: chunk.metadata["language"] = "en"
        except:
            chunk.metadata["language"] = "unknown"
    return chunks

def update_vector_store(force_reload=False):
    """Run the full ingestion pipeline and replace the Chroma DB."""
    print("Starting PDF ingestion...")
    import json
    with open(Path("data/topics.json"), "r", encoding="utf-8") as f:
        topics_json = json.load(f)
    if CHROMA_DB_DIR.exists() and force_reload:
        shutil.rmtree(CHROMA_DB_DIR)
    
    raw_docs = extract_text_from_pdfs()
    chunks = split_documents(raw_docs)
    chunks = add_language_metadata(chunks)
    chunks = assign_topic_metadata(chunks, topics_json)

    # Use the globally defined `embeddings` (set by LLM_PROVIDER)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,           # <--- changed from hardcoded OpenAIEmbeddings
        persist_directory=str(CHROMA_DB_DIR)
    )
    vectordb.persist()
    print(f"Ingestion complete. {len(chunks)} chunks stored.")
    return vectordb


def build_topics_json():
    """Create data/topics.json – {filename: [ {id, title, page}, ... ]}"""
    import json
    topics = {}
    for pdf_file in PDF_DIR.glob("*.pdf"):
        doc = fitz.open(pdf_file)
        toc = doc.get_toc()    # [[level, title, page], ...]
        doc.close()
        if not toc:
            # fallback: use file name as single "topic"
            topics[pdf_file.name] = [{"id": "1", "title": pdf_file.stem, "page": 1}]
            continue
        # build flat list from TOC (only level 1 or 2 headings)
        chapter_list = []
        for i, (level, title, page) in enumerate(toc, start=1):
            # keep only top-level headings; adjust as needed
            if level <= 2:
                chapter_list.append({"id": str(i), "title": title.strip(), "page": page})
        topics[pdf_file.name] = chapter_list
    with open(Path("data/topics.json"), "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2)
    print("Saved topics.json")

def assign_topic_metadata(docs, topics_json):
    for doc in docs:
        source = doc.metadata.get("source", "")
        book_topics = topics_json.get(source, [])
        page = doc.metadata.get("page", 1)
        assigned_topic = None
        # topics already sorted by page, but ensure
        book_topics_sorted = sorted(book_topics, key=lambda x: x["page"])
        for t in book_topics_sorted:
            if page >= t["page"]:
                assigned_topic = t
        if assigned_topic:
            doc.metadata["topic_title"] = assigned_topic["title"]
            doc.metadata["topic_id"] = assigned_topic["id"]
    return docs