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
    user = user_data.setdefault(phone, {"language": "unknown", "provider": provider})
    lang = user["language"]

    # Admin refresh
    if text.startswith("/refresh"):
        parts = text.split(maxsplit=1)
        if len(parts) != 2 or parts[1] != os.getenv("ADMIN_SECRET", "change_me_123"):
            send_message(phone, "Unauthorized.", provider)
            return
        threading.Thread(target=refresh_knowledge, args=(phone, provider)).start()
        return

    # Language selection
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

    # Legal query
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
        global vectorstore, rag_chain
        # Reload the vectorstore
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        rag_chain = build_rag_chain(vectorstore)
        send_message(phone, "✅ Knowledge base refreshed.", provider)
    except Exception as e:
        print("Refresh error:", e)
        send_message(phone, "❌ Refresh failed.", provider)

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
        return PlainTextResponse("OK")
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
    return PlainTextResponse("OK")

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
        return PlainTextResponse("OK")
    handle_message(from_number, text, provider="twilio")
    return PlainTextResponse("OK")