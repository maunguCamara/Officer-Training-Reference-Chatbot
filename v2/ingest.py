import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import fitz  # pymupdf
from langdetect import detect, DetectorFactory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Ensure consistent language detection results
DetectorFactory.seed = 0

# Paths
PDF_DIR = Path("data/pdfs")
CHROMA_DB_DIR = Path("data/chroma_db")

# 1. Extract text from PDFs
def extract_text_from_pdfs(pdf_dir):
    all_documents = []
    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"Processing {pdf_file.name}...")
        doc = fitz.open(pdf_file)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")  # extract as plain text
            text = text.strip()
            if not text:
                continue
            # Add basic cleaning: collapse multiple spaces/newlines
            text = " ".join(text.split())
            # Metadata
            metadata = {
                "source": pdf_file.name,
                "page": page_num + 1  # human‑readable 1‑based
            }
            all_documents.append(
                Document(page_content=text, metadata=metadata)
            )
        doc.close()
    print(f"Extracted {len(all_documents)} page-level documents.")
    return all_documents

# 2. Split into smaller chunks
def split_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    return chunks

# 3. Detect language for each chunk (adds 'language' to metadata)
def add_language_metadata(chunks):
    for chunk in chunks:
        try:
            lang = detect(chunk.page_content)
            # Normalise to two-letter code
            if lang.startswith("en"):
                chunk.metadata["language"] = "en"
            elif lang.startswith("sw"):
                chunk.metadata["language"] = "sw"
            else:
                # Fallback to English if unsure
                chunk.metadata["language"] = "en"
        except:
            chunk.metadata["language"] = "unknown"
    en_count = sum(1 for c in chunks if c.metadata["language"] == "en")
    sw_count = sum(1 for c in chunks if c.metadata["language"] == "sw")
    print(f"Language detection: {en_count} English, {sw_count} Swahili chunks.")
    return chunks

# 4. Embed and store in Chroma
def create_vector_store(chunks, persist_dir):
    # Use OpenAI's multilingual embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Delete any previous DB to start fresh (optional)
    # if persist_dir.exists():
    #     import shutil
    #     shutil.rmtree(persist_dir)
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir)
    )
    vectordb.persist()
    print(f"Vector store persisted at {persist_dir}")
    return vectordb

if __name__ == "__main__":
    # Run pipeline
    raw_docs = extract_text_from_pdfs(PDF_DIR)
    chunks = split_documents(raw_docs)
    chunks = add_language_metadata(chunks)
    vectordb = create_vector_store(chunks, CHROMA_DB_DIR)
    
    # Quick test: retrieve a query in English
    print("\n--- Testing English retrieval ---")
    results = vectordb.similarity_search("What are the grounds for divorce?", k=2)
    for doc in results:
        print(f"[{doc.metadata['source']} p.{doc.metadata['page']} ({doc.metadata['language']})] {doc.page_content[:120]}...")
    
    # Quick test: Swahili query
    print("\n--- Testing Swahili retrieval ---")
    results_sw = vectordb.similarity_search("Ni nini sababu za talaka?", k=2)
    for doc in results_sw:
        print(f"[{doc.metadata['source']} p.{doc.metadata['page']} ({doc.metadata['language']})] {doc.page_content[:120]}...")