import os
import json
import requests
import textwrap
from pathlib import Path
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

# ========== Provider Setup ==========
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

if LLM_PROVIDER == "ollama":
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Using Ollama + HuggingFace embeddings (free).")
else:
    
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
# This global will be reassigned on refresh


def build_rag_chain(vstore):
    """Rebuild the rag_chain with a new vectorstore."""
    new_retriever = vstore.as_retriever(search_kwargs={"k": 4})
    return create_retrieval_chain(new_retriever, question_answer_chain)

def ask_question(query: str, user_language: str):
    return rag_chain.invoke({"input": query, "user_language": user_language})

def ask_with_context(phone: str, query: str) -> str:
    """
    Retrieval‑augmented answer that respects the user’s selected book and topic.
    Filters the vectorstore to only chunks from that book/topic.
    """
    user = user_data.get(phone, {})
    book = user.get("selected_book")
    topic_title = user.get("selected_topic", {}).get("title") if user.get("selected_topic") else None
    lang = user.get("language", "en")

    # Build filter dict for Chroma
    filter_dict = {}
    if book:
        filter_dict["source"] = book             # filename metadata
    if topic_title:
        filter_dict["topic_title"] = topic_title  # assigned during ingestion

    # Create a temporary retriever with the filter
    filtered_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4, "filter": filter_dict} if filter_dict else {"k": 4}
    )

    # Build a new chain on‑the‑fly (cheap, no network call)
    chain = create_retrieval_chain(filtered_retriever, question_answer_chain)
    result = chain.invoke({"input": query, "user_language": lang})
    return result["answer"]

print("Vector store and QA chain ready.")

app = FastAPI()

# ========== User data (in-memory) ==========
user_data = {}

def set_state(phone, state):
    user_data.setdefault(phone, {})["state"] = state

def get_state(phone):
    return user_data.get(phone, {}).get("state", "language_selection")


def get_user_language(phone: str) -> str:
    return user_data.get(phone, {}).get("language", "unknown")

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

def send_message(to: str, text: str, provider: str):
    if provider == "twilio":
        send_twilio_message(to, text)
    else:
        send_meta_message(to, text)

# ========== Core Message Handler ==========
def handle_message(phone: str, text: str, provider: str):
    user = user_data.setdefault(phone, {"language": "en", "state": "language_selection"})
    lang = user.get("language", "en")

    #Answer question with general knowledge
    def is_possible_question(text: str) -> bool:
        """Heuristic: text that is not a simple menu number and is long enough, or ends with ?"""
        t = text.strip()
        if not t:
            return False
        # Single digits 0-9 usually menu choices
        if t.isdigit() and len(t) <= 2:
            return False
        if t.endswith('?') or len(t.split()) >= 3:
            return True
        return False

    # After language is known (state != language_selection), detect global questions
        if user.get("language") != "unknown" and user.get("state") != "language_selection":
            if is_possible_question(text) and text.lower() not in ("menu", "0"):
                # Answer globally across all books
                user["state"] = "chatting"
                user.pop("selected_book", None)
                user.pop("selected_topic", None)
                answer = ask_with_context(phone, text)   # no filter → all books
                disclaimer = get_localized("disclaimer", lang)
                send_long_message(phone, answer + disclaimer, provider)
                return


    # Global commands (work at any time)
    if text.strip().lower() == "menu":
        if user.get("selected_book"):
            # show topic list for the book again
            show_topic_list(phone, provider)
            return
        else:
            show_book_list(phone, provider)
            return

        # Universal back command
    if text.strip() == "0":
        current_state = user.get("state")
        if current_state == "book_selection":
            # go back to language selection
            user["state"] = "language_selection"
            user.pop("selected_book", None)
            show_language_selection(phone, provider)   # implement this
        elif current_state == "topic_selection":
            user["state"] = "book_selection"
            user.pop("selected_book", None)
            show_book_list(phone, provider)
        elif current_state == "chatting":
            user["state"] = "topic_selection"
            # stay on same book, show topic list again
            show_topic_list(phone, provider, user.get("selected_book"))
        else:
            # fallback: restart from language
            user["state"] = "language_selection"
            show_language_selection(phone, provider)
        return

    # State machine
    state = user.get("state", "language_selection")

    if state == "language_selection":
        if text in ("1","2","3","4","5"):   # adjust to your language list
            language_map = {"1":"en", "2":"sw", "3":"fr", "4":"de", "5":"pt"}  # example
            chosen = language_map.get(text, "en")
            user["language"] = chosen
            user["state"] = "book_selection"
            # Reply in chosen language
            reply = get_localized("welcome_book_selection", lang)  # "Please pick a book:"
            show_book_list(phone, provider)
        else:
            # re‑prompt language
            send_message(phone, get_localized("choose_language", lang), provider)
        return

    elif state == "book_selection":
        # user sends a number for a book
        num = text.strip()
        books = list(topics.keys())   # topics loaded from JSON
        try:
            idx = int(num) - 1
            book = books[idx]
            user["selected_book"] = book
            user["state"] = "topic_selection"
            show_topic_list(phone, provider, book)
        except (ValueError, IndexError):
            # Not a valid book number → try as a global question
            if is_possible_question(num):
                user["state"] = "chatting"
                user.pop("selected_book", None)
                user.pop("selected_topic", None)
                answer = ask_with_context(phone, num)
                disclaimer = get_localized("disclaimer", lang)
                send_long_message(phone, answer + disclaimer, provider)
            else:
                send_long_message(phone, get_localized("invalid_choice", lang), provider)
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
            user["selected_topic"] = topic
            user["state"] = "chatting"
            send_topic_summary(phone, provider, book, topic)
        except (ValueError, IndexError):
            if is_possible_question(num):
                user["state"] = "chatting"
                user.pop("selected_topic", None)
                # Keep selected_book? For now, use global (clear it)
                user.pop("selected_book", None)
                answer = ask_with_context(phone, num)
                disclaimer = get_localized("disclaimer", lang)
                send_long_message(phone, answer + disclaimer, provider)
            else:
                send_long_message(phone, get_localized("invalid_choice", lang), provider)
        return

    elif state == "chatting":
        # Free‑text RAG with book/topic filter
        answer = ask_with_context(phone, text)   # filter by book + topic
        disclaimer = get_localized("disclaimer", lang)
        send_long_message(phone, answer + disclaimer, provider)
        return

def send_long_message(phone: str, text: str, provider: str, max_chars: int = 500):
    """Send text in chunks of max_chars, splitting exactly at character boundaries."""
    # Remove leading/trailing whitespace
    text = text.strip()
    total_length = len(text)
    parts = (total_length // max_chars) + (1 if total_length % max_chars else 0)

    for i in range(0, total_length, max_chars):
        chunk = text[i : i + max_chars].strip()
        # Add a small continuation note if there’s more
        if i + max_chars < total_length:
            chunk += f"\n\n({i//max_chars + 1}/{parts})"
        send_message(phone, chunk, provider)
    print(f"Sending chunk of length {len(chunk)}")

LOCALIZED = {
    "en": {
        "choose_language": "Please choose your language:\n1. English\n2. Kiswahili\n3. Pukuti\n4. Français\n5. Deutsch",
        "welcome_book_selection": "Great! Here are the available books:",
        "topic_prompt": "Pick a topic:",
        "disclaimer": "\n\n---\n*This is not legal advice...*",
    },
    "sw": {
        "choose_language": "Tafadhali chagua lugha:\n1. English\n2. Kiswahili\n...",
        "welcome_book_selection": "Sawa! Hapa kuna vitabu:",
        "topic_prompt": "Chagua mada:",
        "disclaimer": "\n\n---\n*Huu sio ushauri wa kisheria...*",
    }
}

def get_localized(key: str, lang: str = "en") -> str:
    """Simple translation lookup. Extend as needed."""
    translations = {
        "en": {
            "choose_language": "Please choose your language:\n1. English\n2. Kiswahili",
            "welcome_book_selection": "Great! Here are the available books:",
            "book_prompt": "Reply with the number of the book you want to explore.",
            "topic_prompt": "Choose a topic by number:",
            "disclaimer": "\n\n---\n*This is not legal advice...*",
            "invalid_choice": "Invalid choice. Please try again.",
            "menu": "Type *menu* to see the topic list again.",
            "no_books": "No books available yet."
        },
        "sw": {
            "choose_language": "Tafadhali chagua lugha:\n1. English\n2. Kiswahili",
            "welcome_book_selection": "Sawa! Haya ndiyo vitabu vinavyopatikana:",
            "book_prompt": "Jibu kwa nambari ya kitabu unachotaka kukisoma.",
            "topic_prompt": "Chagua mada kwa nambari:",
            "disclaimer": "\n\n---\n*Huu sio ushauri wa kisheria...*",
            "invalid_choice": "Chaguo si sahihi. Tafadhali jaribu tena.",
            "menu": "Andika *menu* ili kuona orodha ya mada tena.",
            "no_books": "Hakuna vitabu bado."
        }
    }
    return translations.get(lang, translations["en"]).get(key, key)

def show_book_list(phone: str, provider: str):
    """Send a numbered list of available books (from topics.json keys)."""
    books = list(topics.keys())
    if not books:
        send_long_message(phone, get_localized("no_books", lang), provider)
        return

    lang = user_data.get(phone, {}).get("language", "en")
    # Use the filename (without .pdf) as display name – could be overridden with a translation map
    book_list_lines = [f"{i+1}. {Path(b).stem}" for i, b in enumerate(books)]
    text = get_localized("welcome_book_selection", lang) + "\n" + "\n".join(book_list_lines) + "\n\n" + get_localized("book_prompt", lang)
    send_long_message(phone, text, provider)

def show_topic_list(phone: str, provider: str, book: str = None):
    """Send a numbered list of topics for the given book (or the user's selected book)."""
    if book is None:
        book = user_data.get(phone, {}).get("selected_book")
    if not book or book not in topics:
        send_long_message(phone, get_localized("invalid_choice", lang), provider)
        return

    lang = user_data.get(phone, {}).get("language", "en")
    topic_list = topics[book]
    if not topic_list:
        send_long_message(phone, "This book has no chapters.", provider)
        return

    lines = [f"{t['id']}. {t['title']}" for t in topic_list]
    text = get_localized("topic_prompt", lang) + "\n" + "\n".join(lines) + "\n\n" + get_localized("menu", lang)
    send_long_message(phone, text, provider)

def send_topic_summary(phone: str, provider: str, book: str, topic: dict):
    """Fetch the first few paragraphs for a topic and generate a summary + suggested questions."""
    lang = user_data.get(phone, {}).get("language", "en")
    topic_title = topic["title"]

    # Build a filtered retriever to get context for this topic
    filter_dict = {"source": book, "topic_title": topic_title}
    filtered_retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": filter_dict})
    docs = filtered_retriever.invoke(topic_title)  # retrieve relevant chunks

    # If no docs, fallback to a simple message
    if not docs:
        send_long_message(phone, f"📚 *{topic_title}*\n\n(No content found for this topic yet.)", provider)
        return

    # Build prompt for summary + suggested questions
    context = "\n\n".join([d.page_content for d in docs])
    prompt = (
        f"Based on the following content from the book '{book}', topic '{topic_title}', "
        f"provide a helpful summary in one paragraph (2-3 sentences) in {lang}. "
        f"Then list 5 numbered questions a reader would likely ask about this topic, also in {lang}.\n\n"
        f"Context:\n{context}\n\n"
        f"Output format:\nSummary: ...\n\nQuestions:\n1. ...\n2. ..."
    )

    try:
        # Use the LLM directly (no retriever needed, context already provided)
        result = llm.invoke(prompt)
        answer = result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        print("Summary generation error:", e)
        answer = f"📚 *{topic_title}*\n\n(A problem occurred while generating the summary.)"

    # Add a footer with instructions
    footer = get_localized("menu", lang)
    full_text = answer + "\n\n" + footer
    send_long_message(phone, full_text, provider) 
   

def refresh_knowledge(phone, provider):
    from ingestion import update_vector_store
    try:
        update_vector_store(force_reload=True)
        global vectorstore, rag_chain
        # Reload the vectorstore
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        rag_chain = build_rag_chain(vectorstore)
        send_message(phone, "✅ Knowledge base refreshed.", provider)
    except Exception as e:
        print("Refresh error:", e)
        send_message(phone, "❌ Refresh failed.", provider)

def show_language_selection(phone: str, provider: str):
    """Send the language choice list and set state to language_selection."""
    user_data.setdefault(phone, {})["state"] = "language_selection"
    # You can store a default language or keep existing
    lang = user_data[phone].get("language", "en")
    text = get_localized("choose_language", lang)
    send_long_message(phone, text, provider)
    
# ========== Webhook Endpoints ==========
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
                        user_data.setdefault(from_number, {})["language"] = lang
                        lang_name = "English" if lang == "en" else "Kiswahili"
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