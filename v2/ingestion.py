# ingestion.py
import os
import re
import shutil
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import json
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
    print("Starting PDF ingestion...")
    if CHROMA_DB_DIR.exists() and force_reload:
        shutil.rmtree(CHROMA_DB_DIR)

    # 1. Build / refresh topics.json first
    build_topics_json()                         # creates data/topics.json
    with open(Path("data/topics.json"), "r", encoding="utf-8") as f:
        topics_json = json.load(f)

    # 2. Extract and chunk documents
    raw_docs = extract_text_from_pdfs()
    chunks = split_documents(raw_docs)

    # 3. Add language and topic metadata
    chunks = add_language_metadata(chunks)
    chunks = assign_topic_metadata(chunks, topics_json)   # now topics_json exists

    # 4. Embed and store in Chroma
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,          # uses the global 'embeddings' (free/openai)
        persist_directory=str(CHROMA_DB_DIR)
    )
    vectordb.persist()
    print(f"Ingestion complete. {len(chunks)} chunks stored.")
    return vectordb

def extract_headings_from_pdf(pdf_path):
    """Fallback heading extraction using regex when no TOC exists."""
    doc = fitz.open(pdf_path)
    headings = []
    seen_titles = set()
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        # match lines like "1. Short title and commencement."
        matches = re.findall(r"^(\d+\.\s+[A-Za-z][^\n]{2,})", text, re.MULTILINE)
        for match in matches:
            title = match.strip()
            if title not in seen_titles:
                seen_titles.add(title)
                headings.append((page_num + 1, title))
    doc.close()
    # sort by page (ascending), then by the leading number if present
    headings.sort(key=lambda x: (x[0], int(re.match(r"(\d+)", x[1]).group(1)) if re.match(r"(\d+)", x[1]) else 0))
    # assign ids sequentially, but keep original number for display (already in title)
    chapter_list = [{"id": str(i+1), "title": title, "page": page} for i, (page, title) in enumerate(headings)]
    return chapter_list

def build_topics_json():
    """Create data/topics.json using Part extraction for structured PDFs, fallback otherwise."""
    topics = {}
    for pdf_file in PDF_DIR.glob("*.pdf"):
        doc = fitz.open(pdf_file)
        toc = doc.get_toc()
        doc.close()

        # Try Part extraction first (for legal documents)
        parts = extract_part_topics_from_pdf(pdf_file)
        if parts:
            topics[pdf_file.name] = parts
        elif toc:
            # Use TOC if available
            chapter_list = []
            for i, (level, title, page) in enumerate(toc, start=1):
                if level <= 2:
                    chapter_list.append({"id": str(i), "title": title.strip(), "page": page})
            topics[pdf_file.name] = chapter_list
        else:
            # Fallback: regex heading extraction (your existing code)
            headings = extract_headings_from_pdf(pdf_file)
            topics[pdf_file.name] = headings if headings else [{"id": "1", "title": pdf_file.stem, "page": 1}]
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

def roman_to_int(s):
    """Convert a Roman numeral string to an integer."""
    s = s.upper()
    rom_val = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
    total = 0
    prev = 0
    for ch in reversed(s):
        val = rom_val.get(ch, 0)
        if val >= prev:
            total += val
            prev = val
        else:
            total -= val
    return total

def extract_part_topics_from_pdf(pdf_path):
    """
    Extract Part headings from the PDF.
    Returns a list of dicts: [{"id": "1", "title": "Part I: PRELIMINARY", "page": 13}, ...]
    """
    doc = fitz.open(pdf_path)
    part_entries = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        # Look for lines like "PART I—PRELIMINARY" or "PART II—SAFEGUARDS ..."
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Match pattern: "PART" followed by Roman numeral and then a dash/em-dash
            m = re.match(r'PART\s+([IVXLCDM]+)\s*[—–-]\s*(.*)', line, re.IGNORECASE)
            if m:
                roman = m.group(1)
                title = m.group(2).strip()
                # Convert Roman numeral to integer for ordering
                part_num = roman_to_int(roman)
                # Clean title: remove any trailing section references
                # (e.g., "PRELIMINARY 1—Short title.") -> just "PRELIMINARY"
                # Take only the part before the first section number
                clean_title = re.split(r'\s+\d+\s*[—–-]', title)[0].strip()
                # Fallback: use original title if clean_title is empty
                if not clean_title:
                    clean_title = title.split('—')[0] if '—' in title else title
                # Use the page number of the Part heading (1‑based)
                page = page_num + 1
                # Avoid duplicates (sometimes the same Part appears on multiple pages)
                if not any(e['id'] == str(part_num) for e in part_entries):
                    part_entries.append({
                        "id": str(part_num),
                        "title": f"Part {roman}: {clean_title}",
                        "page": page
                    })
    doc.close()
    # Sort by page number
    part_entries.sort(key=lambda x: x["page"])
    # Re‑assign IDs based on order
    for i, entry in enumerate(part_entries, start=1):
        entry["id"] = str(i)
    return part_entries