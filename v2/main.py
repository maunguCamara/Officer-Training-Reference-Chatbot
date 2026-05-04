import os
import time
import json
import requests
import textwrap
import shutil
from pathlib import Path
from fastapi import FastAPI, Request, Query, Form, File, UploadFile
from fastapi.responses import PlainTextResponse, HTMLResponse
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
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings



import threading
from cachetools import TTLCache

# --- NEW: Database Module ---
import database

load_dotenv()

# ========== Provider Setup ==========
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
USE_E5 = os.getenv("USE_E5", "false").lower() == "true"   # set to 'true' for better multilingual embeddings

if LLM_PROVIDER == "ollama":
    from langchain_ollama import ChatOllama
    if USE_E5:
        from sentence_transformers import SentenceTransformer
        class E5Embeddings:
            def __init__(self):
                self.model = SentenceTransformer("intfloat/multilingual-e5-large")
            def embed_documents(self, texts):
                return self.model.encode(["passage: " + t for t in texts], normalize_embeddings=True)
            def embed_query(self, text):
                return self.model.encode("query: " + text, normalize_embeddings=True)
        embeddings = E5Embeddings()
        print("✅ Using Ollama + E5 multilingual embeddings (free, high quality).")
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("✅ Using Ollama + HuggingFace embeddings (free).")
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
else:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    print("✅ Using OpenAI LLM and embeddings.")

# ========== Config ==========
ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN", "my_custom_verify_token_123")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "change_me_123")
CHROMA_DB_DIR = "data/chroma_db"

twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
twilio_number = os.getenv("TWILIO_WHATSAPP_NUMBER")

pending_feedback = {}
# ========== LangChain Setup ==========
print("Loading vector store...")
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

TOPICS_PATH = Path("data/topics.json")
if TOPICS_PATH.exists():
    with open(TOPICS_PATH, "r", encoding="utf-8") as f:
        topics = json.load(f)
else:
    topics = {}
    print("⚠️ topics.json not found – book/topic menus will be empty.")

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

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def build_rag_chain(vstore):
    new_retriever = vstore.as_retriever(search_kwargs={"k": 4})
    return create_retrieval_chain(new_retriever, question_answer_chain)

# --- Caching ---
answer_cache = TTLCache(maxsize=200, ttl=600)  # 200 items, 10 minutes

def ask_with_context(phone: str, query: str) -> str:
    if ingestion_in_progress:
        return "🛠️ The knowledge base is being rebuilt. Please wait a few minutes and try again."
    user = database.get_user(phone)
    if not user:
        return "User not found. Please restart with /start."
    book = user.get("selected_book")
    topic_title = user.get("selected_topic")  # stored as JSON string? We'll handle as text
    if topic_title:
        try:
            topic_dict = json.loads(topic_title)
            topic_title = topic_dict.get("title", "")
        except:
            pass
    lang = user.get("language", "en")

    cache_key = f"{lang}:{book}:{topic_title}:{query}"
    if cache_key in answer_cache:
        print("CACHE HIT")
        return answer_cache[cache_key]

    conditions = []
    if book:
        conditions.append({"source": {"$eq": book}})
    if topic_title:
        conditions.append({"topic_title": {"$eq": topic_title.strip()}})

    filter_where = None
    if len(conditions) == 1:
        filter_where = conditions[0]
    elif len(conditions) > 1:
        filter_where = {"$and": conditions}

    search_kwargs = {"k": 4}
    if filter_where:
        search_kwargs["filter"] = filter_where
    filtered_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    chain = create_retrieval_chain(filtered_retriever, question_answer_chain)
    result = chain.invoke({"input": query, "user_language": lang})
    answer = result["answer"]
    answer_cache[cache_key] = answer
    return answer

# Initialize database
database.init_db()

print("Vector store and QA chain ready.")

app = FastAPI()

# ========== Messaging Functions ==========
def send_meta_message(to: str, text: str):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
    data = {
        "messaging_product": "whatsapp", "recipient_type": "individual",
        "to": to, "type": "text", "text": {"body": text}
    }
    resp = requests.post(url, json=data, headers=headers)
    if resp.status_code != 200:
        print("Meta send error:", resp.json())

def send_meta_buttons(to: str, body_text: str, buttons: list):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
    btns = [{"type": "reply", "reply": b} for b in buttons]
    data = {
        "messaging_product": "whatsapp", "recipient_type": "individual",
        "to": to, "type": "interactive",
        "interactive": {
            "type": "button", "body": {"text": body_text},
            "action": {"buttons": btns}
        }
    }
    resp = requests.post(url, json=data, headers=headers)
    if resp.status_code != 200:
        print("Meta buttons error:", resp.json())

def send_twilio_message(to: str, text: str):
    try:
        twilio_client.messages.create(
            body=text, from_=twilio_number, to=f"whatsapp:{to}"
        )
    except Exception as e:
        print("Twilio send error:", e)

def send_telegram_message(chat_id: str, text: str):
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not set, cannot send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code != 200:
            print("Telegram send error:", resp.json())
    except Exception as e:
        print("Telegram send exception:", e)

def send_message(to: str, text: str, provider: str):
    print(f"send_message called, provider={provider}, to={to}, text={text[:50]}")
    if provider == "twilio":
        send_twilio_message(to, text)
    elif provider == "telegram":
        send_telegram_message(to, text)
    else:
        send_meta_message(to, text)

# --- Telegram polling ---
def telegram_polling():
    print("🚀 Telegram polling started")
    offset = 0
    while True:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset={offset}&timeout=60"
        try:
            resp = requests.get(url, timeout=65)
            data = resp.json()
            if data["ok"] and data["result"]:
                for upd in data["result"]:
                    offset = upd["update_id"] + 1
                    msg = upd.get("message")
                    if msg and "text" in msg:
                        chat_id = str(msg["chat"]["id"])
                        text = msg["text"]
                        handle_message(chat_id, text, provider="telegram")
        except Exception as e:
            print("Polling error:", e)
        time.sleep(1)

@app.on_event("startup")
def start_polling():
    if TELEGRAM_BOT_TOKEN:
        threading.Thread(target=telegram_polling, daemon=True).start()

# ========== Localization & Footer ==========
def add_footer(text: str, lang: str = "en") -> str:
    footer_en = "\n\n---\n0: back | topics: current book | books: all books"
    footer_sw = "\n\n---\n0: rudi | topics: vitabu hivi | books: vitabu vyote"
    footer = footer_sw if lang == "sw" else footer_en
    return text + footer

LOCALIZED = {
    "en": {
        "choose_language": "Please choose your language:\n1. English\n2. Kiswahili\n3. Pukuti\n4. Français\n5. Deutsch",
        "welcome_book_selection": "Here are the available books:",
        "topic_prompt": "Pick a topic:",
        "disclaimer": "\n\n---\n*This is not legal advice...*",
        "menu": "Type *menu* to see topics again, or *0* to go back.",
        "feedback_prompt": "Was this helpful? Reply /feedback yes or /feedback no",
        "feedback_invalid": "Invalid. Reply /feedback yes or /feedback no",
        "feedback_thanks": "Thank you for your feedback!",
        "search_prompt": "Usage: /search your question",
        "search_no_results": "No matching content found.",
    },
    "sw": {
        "choose_language": "Tafadhali chagua lugha:\n1. English\n2. Kiswahili\n...",
        "welcome_book_selection": "Sawa! Hapa kuna vitabu:",
        "topic_prompt": "Chagua mada:",
        "disclaimer": "\n\n---\n*Huu sio ushauri wa kisheria...*",
        "menu": "Andika *menu* ili kuona mada tena, au *0* kurudi nyuma.",
        "feedback_prompt": "Je, hii ilikusaidia? Jibu /feedback ndio au /feedback la",
        "feedback_invalid": "Batili. Jibu /feedback ndio au /feedback la",
        "feedback_thanks": "Asante kwa maoni yako!",
        "search_prompt": "Matumizi: /search swali lako",
        "search_no_results": "Hakuna maudhui yanayolingana.",
    }
}

def get_localized(key: str, lang: str = "en") -> str:
    # Shorter, often used translations (same as before)
    translations = {
        "en": {
            "choose_language": "Please choose your language:\n1. English\n2. Kiswahili",
            "welcome_book_selection": "Great! Here are the available books:",
            "book_prompt": "Reply with the number of the book you want to explore.",
            "topic_prompt": "Choose a topic by number:",
            "disclaimer": "\n\n---\n*This is not legal advice...*",
            "invalid_choice": "Invalid choice. Please try again.",
            "menu": "Type *menu* to see the topic list again.",
            "no_books": "No books available yet.",
            "feedback_prompt": "Was this helpful? Reply /feedback yes or /feedback no",
            "feedback_invalid": "Invalid. Reply /feedback yes or /feedback no",
            "feedback_thanks": "Thank you for your feedback!",
            "search_prompt": "Usage: /search your question",
            "search_no_results": "No matching content found.",
        },
        "sw": {
            "choose_language": "Tafadhali chagua lugha:\n1. English\n2. Kiswahili",
            "welcome_book_selection": "Sawa! Haya ndiyo vitabu vinavyopatikana:",
            "book_prompt": "Jibu kwa nambari ya kitabu unachotaka kukisoma.",
            "topic_prompt": "Chagua mada kwa nambari:",
            "disclaimer": "\n\n---\n*Huu sio ushauri wa kisheria...*",
            "invalid_choice": "Chaguo si sahihi. Tafadhali jaribu tena.",
            "menu": "Andika *menu* ili kuona orodha ya mada tena.",
            "no_books": "Hakuna vitabu bado.",
            "feedback_prompt": "Je, hii ilikusaidia? Jibu /feedback ndio au /feedback la",
            "feedback_invalid": "Batili. Jibu /feedback ndio au /feedback la",
            "feedback_thanks": "Asante kwa maoni yako!",
            "search_prompt": "Matumizi: /search swali lako",
            "search_no_results": "Hakuna maudhui yanayolingana.",
        }
    }
    return translations.get(lang, translations["en"]).get(key, key)

def init_vectorstore(force_rebuild=False):
    global vectorstore, retriever, rag_chain, ingestion_in_progress
    ingestion_in_progress = False

    print("Loading vector store...")
    try:
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("✅ Vector store ready.")
    except Exception as e:
        print(f"⚠️ Vector store error: {e}")
        print("🗑️  Deleting corrupted database and rebuilding...")
        # Wipe both Chroma's internal cache and our local folder
        shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
        shutil.rmtree(Path.home() / ".chromadb", ignore_errors=True)

        # Create an empty but valid Chroma store
        dummy_text = ["placeholder"]
        vectorstore = Chroma.from_texts(dummy_text, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Start background re‑ingestion of the real PDFs
        ingestion_in_progress = True
        threading.Thread(target=_background_ingest, daemon=True).start()

def _background_ingest():
    global vectorstore, retriever, rag_chain, ingestion_in_progress
    try:
        from ingestion import update_vector_store
        update_vector_store(force_reload=True)
        # Reload the fresh store
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("✅ Background ingestion complete.")
    except Exception as e:
        print(f"❌ Background ingestion failed: {e}")
    finally:
        ingestion_in_progress = False

# Flag to check while ingestion is in progress
ingestion_in_progress = False
init_vectorstore(force_rebuild=False)

# ========== Core Message Handler ==========
def handle_message(phone: str, text: str, provider: str):
    user = database.get_user(phone)
    if not user:
        user = {"user_id": phone, "language": "en", "state": "language_selection"}
        database.set_user(phone, language="en", state="language_selection")
    lang = user.get("language", "en")

    # --- Global commands (before state machine) ---
    # --- Reset commands (always go to language selection) ---
    if text.strip().lower() in ("/start", "hello", "hi", "habari", "mambo"):
        database.set_user(phone, language="en", state="language_selection", selected_book=None, selected_topic=None)
        user = database.get_user(phone) or {}
        lang = user.get("language", "en")
        show_language_selection(phone, provider)
        return
    # Feedback
    if text.startswith("/feedback"):
        parts = text.split()
        if len(parts) < 2:
            send_message(phone, get_localized("feedback_prompt", lang), provider)
            return
        rating_text = parts[1].lower()
        if rating_text in ["yes", "1", "good", "ndio"]:
            rating = 1
        elif rating_text in ["no", "0", "bad", "la"]:
            rating = 0
        else:
            send_message(phone, get_localized("feedback_invalid", lang), provider)
            return
        fb = pending_feedback.get(phone, {})
        last_q = fb.get("last_query", "")
        last_a = fb.get("last_answer", "")
        if not last_q or not last_a:
            send_message(phone, "No recent question to rate. Ask something first.", provider)
            return
        database.save_feedback(phone, last_q, last_a, rating)
        # Clear after saving
        pending_feedback.pop(phone, None)
        send_message(phone, get_localized("feedback_thanks", lang), provider)
        return
    # Search across all books
    if text.strip().lower().startswith("/search"):
        query = text[len("/search"):].strip()
        if not query:
            send_message(phone, get_localized("search_prompt", lang), provider)
            return
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        if not docs:
            send_message(phone, get_localized("search_no_results", lang), provider)
            return
        results = []
        for d in docs:
            book = Path(d.metadata.get("source", "unknown")).stem
            topic = d.metadata.get("topic_title", "N/A")
            page = d.metadata.get("page", "?")
            snippet = d.page_content[:150] + "..."
            results.append(f"📖 {book} – {topic} (p. {page})\n{snippet}")
        combined = "\n\n".join(results)
        send_long_message(phone, combined, provider, lang=lang)
        return

    # Existing global commands
    if text.strip().lower() == "menu":
        if user.get("selected_book"):
            show_topic_list(phone, provider)
        else:
            show_book_list(phone, provider)
        return

    if text.strip().lower() == "topics":
        if user.get("selected_book"):
            show_topic_list(phone, provider, user["selected_book"])
        else:
            send_long_message(phone, get_localized("no_books", lang), provider, lang=lang)
        return

    if text.strip().lower() == "books":
        database.set_user(phone, state="book_selection")
        show_book_list(phone, provider)
        return

    if text.strip() == "0":
        current_state = user.get("state")
        if current_state == "book_selection":
            database.set_user(phone, state="language_selection", selected_book=None)
            show_language_selection(phone, provider)
        elif current_state == "topic_selection":
            database.set_user(phone, state="book_selection", selected_book=None)
            show_book_list(phone, provider)
        elif current_state == "chatting":
            database.set_user(phone, state="topic_selection")
            show_topic_list(phone, provider, user.get("selected_book"))
        else:
            database.set_user(phone, state="language_selection")
            show_language_selection(phone, provider)
        return

    # --- Question detection (bypass state for direct questions) ---
    def is_possible_question(t: str) -> bool:
        t = t.strip()
        if not t:
            return False
        if t.isdigit() and len(t) <= 2:
            return False
        if t.endswith('?') or len(t.split()) >= 3:
            return True
        return False

    if user.get("language") != "unknown" and user.get("state") != "language_selection":
        if is_possible_question(text) and text.lower() not in ("menu", "0"):
            database.set_user(phone, state="chatting", selected_book=None, selected_topic=None)
            answer = ask_with_context(phone, text)
            user = database.get_user(phone)  # refresh after modifications
            disclaimer = get_localized("disclaimer", lang)
            full_reply = answer + disclaimer + "\n\n" + get_localized("feedback_prompt", lang)
            # Store for feedback
            #database.set_user(phone, last_query=text, last_answer=answer)
            send_long_message(phone, full_reply, provider, lang=lang)
            return

    # --- State machine ---
    state = user.get("state", "language_selection")

    if state == "language_selection":
        if text in ("1","2","3","4","5"):
            language_map = {"1":"en", "2":"sw", "3":"fr", "4":"de", "5":"pt"}
            chosen = language_map.get(text, "en")
            database.set_user(phone, language=chosen, state="book_selection")
            show_book_list(phone, provider)
        else:
            send_message(phone, get_localized("choose_language", lang), provider)
        return

    elif state == "book_selection":
        num = text.strip()
        books = list(topics.keys())
        try:
            idx = int(num) - 1
            book = books[idx]
            database.set_user(phone, selected_book=book, state="topic_selection")
            database.save_analytics("book_selected", phone, book=book)
            show_topic_list(phone, provider, book)
        except (ValueError, IndexError):
            if is_possible_question(num):
                database.set_user(phone, state="chatting", selected_book=None, selected_topic=None)
                answer = ask_with_context(phone, num)
                disclaimer = get_localized("disclaimer", lang)
                full_reply = answer + disclaimer + "\n\n" + get_localized("feedback_prompt", lang)
                #database.set_user(phone, last_query=num, last_answer=answer)
                send_long_message(phone, full_reply, provider, lang=lang)
            else:
                send_long_message(phone, get_localized("invalid_choice", lang), provider, lang=lang)
        return

    elif state == "topic_selection":
        num = text.strip()
        book = user.get("selected_book")
        if not book:
            send_message(phone, "Error. Type menu.", provider)
            return
        topics_list = topics.get(book, [])
        try:
            idx = int(num) - 1
            topic = topics_list[idx]
            # Store selected_topic as JSON string for simplicity
            database.set_user(phone, selected_topic=json.dumps(topic), state="chatting")
            database.save_analytics("topic_selected", phone, book=book, topic_title=topic.get("title"))
            send_topic_summary(phone, provider, book, topic)
        except (ValueError, IndexError):
            if is_possible_question(num):
                database.set_user(phone, state="chatting", selected_topic=None)
                answer = ask_with_context(phone, num)
                disclaimer = get_localized("disclaimer", lang)
                full_reply = answer + disclaimer + "\n\n" + get_localized("feedback_prompt", lang)
                #database.set_user(phone, last_query=num, last_answer=answer)
                send_long_message(phone, full_reply, provider, lang=lang)
            else:
                send_long_message(phone, get_localized("invalid_choice", lang), provider, lang=lang)
        return

    elif state == "chatting":
        if text.strip().lower() == "more":
            if user.get("selected_topic") and user.get("selected_book"):
                try:
                    topic_dict = json.loads(user["selected_topic"])
                    send_full_part(phone, provider, user["selected_book"], topic_dict["title"], lang)
                except:
                    send_long_message(phone, "No topic selected. Use 'topics' first.", provider, lang=lang)
            else:
                send_long_message(phone, "No topic selected. Use 'topics' first.", provider, lang=lang)
            return

        # Normal QA
        answer = ask_with_context(phone, text)
        pending_feedback[phone] = {"last_query": text, "last_answer": answer}
        disclaimer = get_localized("disclaimer", lang)
        full_reply = answer + disclaimer + "\n\n" + get_localized("feedback_prompt", lang)
        #database.set_user(phone, last_query=text, last_answer=answer)
        database.save_analytics("question", phone)
        send_long_message(phone, full_reply, provider, lang=lang)
        return

def send_long_message(phone: str, text: str, provider: str, max_chars=500, lang="en"):
    text_with_footer = add_footer(text.strip(), lang)
    total_length = len(text_with_footer)
    parts = (total_length // max_chars) + (1 if total_length % max_chars else 0)
    for i in range(0, total_length, max_chars):
        chunk = text_with_footer[i:i+max_chars].strip()
        if i + max_chars < total_length:
            chunk += f"\n\n({i//max_chars + 1}/{parts})"
        send_message(phone, chunk, provider)

def ussd_router(session_id: str, phone: str, text: str) -> str:
    """Process a USSD request and return the response text (max 182 chars)."""
    session = ussd_sessions.setdefault(session_id, {"state": "main", "language": "en"})
    lang = session["language"]
    user_input = text.strip() if text else ""

    # Helper to prepend CON or END
    def respond(prefix, msg):
        return f"{prefix} {msg}"[:182]

    # Fresh session
    if not user_input:
        return respond("CON", get_localized_ussd("ussd_welcome", lang))

    # Language change
    if user_input.upper() == "LANG":
        session["state"] = "language_selection"
        return respond("CON", get_localized_ussd("ussd_choose_lang", lang))

    if session.get("state") == "language_selection":
        if user_input in ("1", "2"):
            lang_map = {"1": "en", "2": "sw"}
            new_lang = lang_map.get(user_input, "en")
            session["language"] = new_lang
            session["state"] = "main"
            return respond("END", get_localized_ussd("ussd_lang_set", new_lang))
        else:
            return respond("CON", get_localized_ussd("ussd_invalid_lang", lang))

    # Main state – answer query
    answer, source = ask_ussd(user_input, lang)
    full = f"{answer}{source}"
    # Ensure total length ≤ 160 to leave room for "END "
    if len(full) > 155:
        full = full[:152] + "..."
    return respond("END", full)

# --- Book/Topic/Summary functions (same as before, but use database) ---
def show_book_list(phone: str, provider: str):
    user = database.get_user(phone) or {}
    lang = user.get("language", "en")
    books = list(topics.keys())
    if not books:
        send_long_message(phone, get_localized("no_books", lang), provider, lang=lang)
        return
    book_list_lines = [f"{i+1}. {Path(b).stem}" for i, b in enumerate(books)]
    text = get_localized("welcome_book_selection", lang) + "\n" + "\n".join(book_list_lines) + "\n\n" + get_localized("book_prompt", lang)
    send_long_message(phone, text, provider, lang=lang)

def show_topic_list(phone: str, provider: str, book: str = None):
    user = database.get_user(phone) or {}
    lang = user.get("language", "en")
    if book is None:
        book = user.get("selected_book")
    if not book or book not in topics:
        send_long_message(phone, get_localized("invalid_choice", lang), provider, lang=lang)
        return
    topic_list = topics[book]
    if not topic_list:
        send_long_message(phone, "This book has no chapters.", provider, lang=lang)
        return
    lines = [f"{t['id']}. {t['title']}" for t in topic_list]
    text = get_localized("topic_prompt", lang) + "\n" + "\n".join(lines) + "\n\n" + get_localized("menu", lang)
    send_long_message(phone, text, provider, lang=lang)

def send_topic_summary(phone: str, provider: str, book: str, topic: dict):
    user = database.get_user(phone) or {}
    lang = user.get("language", "en")
    topic_title = topic["title"].strip()

    conditions = [
        {"source": {"$eq": book}},
        {"topic_title": {"$eq": topic_title}}
    ]
    filter_where = {"$and": conditions}
    filtered_retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": filter_where})
    docs = filtered_retriever.invoke(topic_title)

    if not docs:
        fallback_retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"source": {"$eq": book}}})
        docs = fallback_retriever.invoke(topic_title)
        if not docs:
            send_long_message(phone, f"📚 *{topic_title}*\n\n(No content found.)", provider, lang=lang)
            return

    context = "\n\n".join([d.page_content for d in docs])
    prompt = (
        f"Based on the following content from the book '{book}', topic '{topic_title}', "
        f"provide a helpful summary in one paragraph (2-3 sentences) in {lang}. "
        f"Then list 5 numbered questions a reader would likely ask about this topic, also in {lang}.\n\n"
        f"Context:\n{context}\n\n"
        f"Output format:\nSummary: ...\n\nQuestions:\n1. ...\n2. ..."
    )
    try:
        result = llm.invoke(prompt)
        answer = result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        print("Summary error:", e)
        answer = f"📚 *{topic_title}*\n\n(Problem generating summary.)"

    instruction = "Type *more* to read the full text of this Part.\n" if lang == "en" else "Andika *more* ili kusoma maandishi yote ya Sehemu hii.\n"
    footer = get_localized("menu", lang)
    full_text = answer + "\n\n" + instruction + footer
    send_long_message(phone, full_text, provider, lang=lang)

def send_full_part(phone: str, provider: str, book: str, topic_title: str, lang: str):
    filter_where = {"$and": [{"source": {"$eq": book}}, {"topic_title": {"$eq": topic_title.strip()}}]}
    retriever = vectorstore.as_retriever(search_kwargs={"k": 100, "filter": filter_where})
    docs = retriever.invoke(topic_title)
    if not docs:
        send_long_message(phone, "(No content found.)", provider, lang=lang)
        return
    docs_sorted = sorted(docs, key=lambda d: d.metadata.get("page", 0))
    content = "\n\n".join([f"Page {d.metadata.get('page', '?')}:\n{d.page_content}" for d in docs_sorted])
    send_long_message(phone, content, provider, lang=lang, max_chars=1000)

def refresh_knowledge(phone, provider):
    from ingestion import update_vector_store
    try:
        update_vector_store(force_reload=True)
        global vectorstore, rag_chain
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        rag_chain = build_rag_chain(vectorstore)
        send_message(phone, "✅ Knowledge base refreshed.", provider)
    except Exception as e:
        print("Refresh error:", e)
        send_message(phone, "❌ Refresh failed.", provider)

def show_language_selection(phone: str, provider: str):
    database.set_user(phone, state="language_selection")
    user = database.get_user(phone) or {}
    lang = user.get("language", "en")
    text = get_localized("choose_language", lang)
    send_long_message(phone, text, provider, lang=lang)

# --- Admin endpoints ---
@app.post("/admin/upload")
async def admin_upload(file: UploadFile = File(...), token: str = Form(...)):
    if token != ADMIN_SECRET:
        return PlainTextResponse("Unauthorized", status_code=401)
    file_path = Path("data/pdfs") / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    threading.Thread(target=refresh_knowledge, args=("admin", "meta")).start()
    return PlainTextResponse("Uploaded. Ingestion started.")

@app.get("/admin/stats", response_class=HTMLResponse)
async def admin_stats(token: str = Query(...)):
    if token != ADMIN_SECRET:
        return HTMLResponse("Unauthorized", status_code=401)
    stats = database.get_stats()
    html = f"""
    <h1>Bot Stats</h1>
    <p>Total users: {stats['total_users']}</p>
    <p>Total queries: {stats['total_queries']}</p>
    <h2>Top Books</h2>
    <ul>
    """ + "\n".join([f"<li>{b['book']}: {b['count']}</li>" for b in stats['top_books']]) + "</ul>"
    return html

@app.get("/health")
async def health_check():
    status = {"database": "ok", "llm": "ok", "vectorstore": "ok"}
    try:
        vectorstore.similarity_search("test", k=1)
    except Exception as e:
        status["vectorstore"] = f"error: {e}"
    try:
        llm.invoke("ping")
    except Exception as e:
        status["llm"] = f"error: {e}"
    return status

# ========== USSD (unchanged, but now uses database) ==========
# (You can refactor USSD similarly to use database for session storage if needed)
# For now, USSD still uses in-memory dictionaries; you can migrate later.
ussd_sessions = {}  # still in memory for USSD
# ... (keep existing USSD functions unchanged) ...

# Webhook endpoints same as before
# ... (keep Meta verify, Meta webhook, Twilio webhook, Telegram webhook) ...
# Note: In Meta webhook and Twilio webhook, replace user_data accesses with database functions.
# I'll show the updated webhooks below.

# ========== Webhook Endpoints ==========
@app.api_route("/webhook/telegram", methods=["GET", "POST"])
async def telegram_webhook(request: Request):
    if request.method == "GET":
        return PlainTextResponse("Telegram webhook is active", status_code=200)
    try:
        body = await request.json()
    except:
        return PlainTextResponse("", status_code=200)
    if "message" not in body:
        return PlainTextResponse("", status_code=200)
    chat_id = str(body["message"]["chat"]["id"])
    text = body["message"].get("text", "")
    if not text:
        return PlainTextResponse("", status_code=200)
    handle_message(chat_id, text, provider="telegram")
    return PlainTextResponse("", status_code=200)

@app.post("/ussd")
async def ussd_endpoint(
    sessionId: str = Form(...),
    phoneNumber: str = Form(...),
    text: str = Form(default="")
):
    response_text = ussd_router(sessionId, phoneNumber, text)
    return PlainTextResponse(response_text)

@app.get("/webhook")
async def meta_verify(hub_mode=Query(alias="hub.mode"), hub_challenge=Query(alias="hub.challenge"),
                      hub_verify_token=Query(alias="hub.verify_token")):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge)
    return PlainTextResponse("Verification failed", status_code=403)

@app.post("/webhook")
async def meta_webhook(request: Request):
    body = await request.json()
    if body.get("object") != "whatsapp_business_account":
        return PlainTextResponse("", status_code=200)
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            if "messages" not in value:
                continue
            for msg in value["messages"]:
                from_number = msg["from"]
                msg_type = msg.get("type")
                if msg_type == "interactive":
                    button_id = msg["interactive"]["button_reply"]["id"]
                    if button_id in ("lang_en", "lang_sw"):
                        lang = "en" if button_id == "lang_en" else "sw"
                        database.set_user(from_number, language=lang, state="book_selection")
                        send_meta_message(from_number, f"Language set to {lang_name}. Ask your legal question.")
                    else:
                        send_meta_message(from_number, "Unknown selection. Type /start to choose language.")
                    continue
                if msg_type == "text":
                    text = msg["text"]["body"].strip()
                    handle_message(from_number, text, provider="meta")
    return PlainTextResponse("", status_code=200)

@app.api_route("/webhook/twilio", methods=["GET", "POST"])
async def twilio_webhook(request: Request):
    if request.headers.get("content-type") == "application/json":
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)
    from_number = body.get("From", "").replace("whatsapp:", "")
    text = body.get("Body", "").strip()
    if not from_number:
        return PlainTextResponse("", status_code=200)
    handle_message(from_number, text, provider="twilio")
    return PlainTextResponse("", status_code=200)