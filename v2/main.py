import os
import requests
from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from twilio.rest import Client

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_classic.prompts import PromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import threading

load_dotenv()

# ========== Provider Setup (free / openai) ==========
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

if LLM_PROVIDER == "ollama":
    # Use ChatOllama for chat model   
    from langchain_ollama import OllamaLLM
    llm = ChatOllama(model="llama3.2:1b", temperature=0)         # Chat model works with ChatPromptTemplate
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Using Ollama + HuggingFace embeddings (free).")
else:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    print("✅ Using OpenAI LLM and embeddings.")

# ========== Config ==========
ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN", "my_custom_verify_token_123")
CHROMA_DB_DIR = "data/chroma_db"

twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
twilio_number = os.getenv("TWILIO_WHATSAPP_NUMBER")

# ========== LangChain Setup (New API) ==========

print("Loading vector store...")
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# System prompt – note we now use {context} (from retriever) and {input} (user query)
# We will pass user_language as part of the input by merging it with the query
system_prompt = (
    "You are a legal training assistant. Your answers must be based ONLY on the provided context.\n"
    "If the context does not contain the answer, say 'Samahani, siwezi kupata jibu kutoka kwenye nyaraka zilizopo.' (Swahili) or "
    "'Sorry, I cannot find the answer in the available documents.' (English) depending on the user's language.\n"
    "Always cite the source file and page number for each statement.\n\n"
    "The user's preferred language is: {user_language}\n"
    "- If the language is English, answer entirely in English.\n"
    "- If the language is Swahili, answer entirely in Swahili.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the "stuff documents" chain (combines retrieved docs with LLM)
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create the full retrieval chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Wrapper function to pass user_language into the chain
def ask_question(query: str, user_language: str):
    """
    Invoke the RAG chain with the user query and language.
    The retriever expects {input}, the prompt expects {input, user_language, context}.
    We use a small runnable passthrough to inject user_language.
    """
    # We'll merge user_language into the input dict so the prompt can see it
    result = rag_chain.invoke({"input": query, "user_language": user_language})
    # result contains "answer" (from create_stuff_documents_chain) and "context" (the docs)
    return result

print("Vector store and QA chain ready.")

app = FastAPI()

# ========== User data (in-memory) ==========
user_data = {}

def get_user_language(phone: str) -> str:
    return user_data.get(phone, {}).get("language", "unknown")

# ========== Messaging (unchanged) ==========
# ... (keep all your existing send_message, send_meta_message, send_twilio_message, etc.)
# I'll just include the handler that uses the new chain.

def handle_message(phone: str, text: str, provider: str):
    user = user_data.setdefault(phone, {"language": "unknown", "provider": provider})
    lang = user["language"]

    if text.startswith("/refresh"):
        parts = text.split(maxsplit=1)
        if len(parts) != 2 or parts[1] != os.getenv("ADMIN_SECRET", "change_me_123"):
            send_message(phone, "Unauthorized.", provider)
            return
        threading.Thread(target=refresh_knowledge_base, args=(phone, provider)).start()
        return

    if lang == "unknown" or text.lower().strip() in ("/start", "hello", "hi", "habari", "mambo"):
        if provider == "meta":
            send_meta_buttons(phone, "Welcome! Please choose your language:",
                              [{"id": "lang_en", "title": "English"}, {"id": "lang_sw", "title": "Kiswahili"}])
        else:
            send_message(phone, "Karibu / Welcome to Legal Bot! Please choose your language:\nReply *1* for English\nReply *2* for Kiswahili", provider)
        return

    if text == "1":
        user["language"] = "en"
        send_message(phone, "Language set to English. Ask your legal question.", provider)
        return
    elif text == "2":
        user["language"] = "sw"
        send_message(phone, "Lugha imewekwa Kiswahili. Uliza swali lako la kisheria.", provider)
        return

    # Legal query – use the new chain
    try:
        result = ask_question(text, lang)
        answer = result["answer"]
        disclaimer = (
            "\n\n---\n*Tahadhari: Hii siyo ushauri wa kisheria. Wasiliana na wakili kwa mwongozo rasmi.*"
            if lang == "sw" else
            "\n\n---\n*Disclaimer: This is not legal advice. Consult a qualified lawyer for official guidance.*"
        )
        full_reply = answer + disclaimer
        if len(full_reply) > 4000:
            full_reply = full_reply[:4000] + "\n... (message truncated)"
        send_message(phone, full_reply, provider)
    except Exception as e:
        print("QA error:", e)
        err_msg = "Samahani, kuna hitilafu." if lang == "sw" else "Sorry, an error occurred."
        send_message(phone, err_msg, provider)

def refresh_knowledge(phone, provider):
    from ingestion import update_vector_store
    try:
        update_vector_store(force_reload=True)
        global vectorstore
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        qa_chain.retriever = retriever
        send_message(phone, "✅ Knowledge base refreshed.", provider)
    except Exception as e:
        print("Refresh error:", e)
        send_message(phone, "❌ Refresh failed.", provider)