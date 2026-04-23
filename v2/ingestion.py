# ingestion.py
import shutil
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import fitz
from langdetect import detect, DetectorFactory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

DetectorFactory.seed = 0

PDF_DIR = Path("data/pdfs")
CHROMA_DB_DIR = Path("data/chroma_db")

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
    if CHROMA_DB_DIR.exists() and force_reload:
        shutil.rmtree(CHROMA_DB_DIR)
    raw_docs = extract_text_from_pdfs()
    chunks = split_documents(raw_docs)
    chunks = add_language_metadata(chunks)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DB_DIR)
    )
    vectordb.persist()
    print(f"Ingestion complete. {len(chunks)} chunks stored.")
    return vectordb