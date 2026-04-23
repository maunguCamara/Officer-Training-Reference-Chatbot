import os
import requests
from fastapi import FastAPI, Request, Query, Form
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from twilio.rest import ClientS
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()
twilio_client = Client(os.getenv("TWILIO_ACCCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
twilio number = os.getenv("TWILIO_WHATSAPP_NUMBER")s

app = FastAPI()

# ---------- Config ----------
ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = "my_custom_verify_token_123"

CHROMA_DB_DIR = "data/chroma_db"

# ---------- Global LLM & vector store ----------
# Load once at startup (can take a few seconds)
print("Loading vector store...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # fetch 4 most relevant chunks

# LLM – use gpt-4o-mini for cheaper, fast responses
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------- Custom Prompt ----------
# The prompt enforces: answer in the user's preferred language, cite sources,
# never make up info, and stay strictly within the context.
template = """You are a legal training assistant. Your answers must be based ONLY on the provided context.
If the context does not contain the answer, say "Samahani, siwezi kupata jibu kutoka kwenye nyaraka zilizopo." (in Swahili) or "Sorry, I cannot find the answer in the available documents." (in English) depending on the user's language.
Always cite the source file and page number for each statement.

The user's preferred language is: {user_language}
- If the user's language is English, answer entirely in English.
- If the user's language is Swahili, answer entirely in Swahili.
- If the user's language is unknown, answer in the same language as the question.

Context:
{context}

Question: {question}
Helpful answer (include source citations):"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question", "user_language"]
)

# Build the RetrievalQA chain – we'll pass "user_language" as extra input per call.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True
)

print("Vector store and QA chain ready.")

# ---------- User language storage (in‑memory) ----------
# In production, replace with Redis, database, or persistent file.
user_language_prefs = {}

def get_user_language(phone_number: str) -> str:
    return user_language_prefs.get(phone_number, "unknown")

# ---------- Meta API Messaging ----------
def send_text_message(to: str, text: str):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }
    resp = requests.post(url, json=data, headers=headers)
    if resp.status_code != 200:
        print("Failed to send text:", resp.json())

def send_interactive_buttons(to: str, body_text: str, buttons: list):
    """
    buttons: list of dicts with "type" ("reply") and "reply" dict with "id" and "title"
    """
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body_text},
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {"id": btn["id"], "title": btn["title"]}
                    } for btn in buttons
                ]
            }
        }
    }
    resp = requests.post(url, json=data, headers=headers)
    if resp.status_code != 200:
        print("Failed to send interactive buttons:", resp.json())

# ---------- Webhook Verification (Week 1 unchanged) ----------
@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(alias="hub.mode"),
    hub_challenge: str = Query(alias="hub.challenge"),
    hub_verify_token: str = Query(alias="hub.verify_token")
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge)
    else:
        return PlainTextResponse("Verification failed", status_code=403)

# ---------- Main Webhook Handler ----------
@app.post("/webhook")
async def receive_message(request: Request):

    body = await request.json()
    # print("Incoming:", json.dumps(body, indent=2))  # debug if needed

    if body.get("object") == "whatsapp_business_account":
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                
                # ---- Handle incoming messages ----
                if "messages" in value:
                    for msg in value["messages"]:
                        from_number = msg["from"]
                        msg_type = msg.get("type")

                        # --- Interactive button reply ---
                        if msg_type == "interactive":
                            interactive_data = msg["interactive"]
                            if interactive_data["type"] == "button_reply":
                                button_id = interactive_data["button_reply"]["id"]
                                if button_id in ("lang_en", "lang_sw"):
                                    user_language_prefs[from_number] = "en" if button_id == "lang_en" else "sw"
                                    lang_name = "English" if button_id == "lang_en" else "Kiswahili"
                                    send_text_message(from_number, f"Language set to {lang_name}. You can now ask your legal question.")
                                else:
                                    send_text_message(from_number, "Unknown selection. Please type /start to choose language.")
                                continue

                        # --- Text message ---
                        elif msg_type == "text":
                            text_body = msg["text"]["body"].strip()

                            # Simple "start" or first contact
                            if text_body.lower() in ["/start", "hello", "hi", "habari", "mambo"]:
                                # Ask for language
                                send_interactive_buttons(
                                    to=from_number,
                                    body_text="Welcome to the Legal Training Bot! Please choose your language:",
                                    buttons=[
                                        {"id": "lang_en", "title": "English"},
                                        {"id": "lang_sw", "title": "Kiswahili"}
                                    ]
                                )
                                continue

                            # If user hasn't set language, prompt them
                            pref_language = get_user_language(from_number)
                            if pref_language == "unknown":
                                send_interactive_buttons(
                                    to=from_number,
                                    body_text="Please choose your preferred language first:",
                                    buttons=[
                                        {"id": "lang_en", "title": "English"},
                                        {"id": "lang_sw", "title": "Kiswahili"}
                                    ]
                                )
                                continue

                            # --- Process legal query ---
                            # Run the QA chain, passing user_language
                            try:
                                result = qa_chain({"query": text_body, "user_language": pref_language})
                                answer = result["result"]
                                sources = result.get("source_documents", [])

                                # Add disclaimer footer
                                disclaimer = (
                                    "\n\n---\n*Tahadhari: Hii siyo ushauri wa kisheria. Wasiliana na wakili kwa mwongozo rasmi.*"
                                    if pref_language == "sw"
                                    else "\n\n---\n*Disclaimer: This is not legal advice. Consult a qualified lawyer for official guidance.*"
                                )
                                full_reply = answer + disclaimer

                                # WhatsApp has a 4096 character limit; split if needed
                                if len(full_reply) > 4000:
                                    # Send truncated with a note
                                    full_reply = full_reply[:4000] + "\n... (message truncated)"
                                send_text_message(from_number, full_reply)

                            except Exception as e:
                                print("QA error:", e)
                                error_msg = (
                                    "Samahani, kuna hitilafu. Jaribu tena baadaye."
                                    if pref_language == "sw"
                                    else "Sorry, an error occurred. Please try again later."
                                )
                                send_text_message(from_number, error_msg)

    return PlainTextResponse("OK", status_code=200)
@app.api_route("/webhook/twilio", methods=["GET", "POST"])
async def twilio_webhook(request: Request):
    """Handle incoming WhatsApp messages from Twilio."""
    # Detect content type: form-encoded or JSON
    if request.headers.get("content-type") == "application/json":
        body = await request.json()
    else:
        form = await request.form()
        # Convert form data to dict
        body = dict(form)

    # Extract essential fields
    from_number = body.get("From", "")
    if from_number.startswith("whatsapp:"):
        # strip prefix for consistency, keep E.164
        from_number = from_number.replace("whatsapp:", "")
    msg_body = body.get("Body", "").strip()
    
    # Ensure user data exists
    if from_number not in user_data:
        user_data[from_number] = {"language": "unknown", "provider": "twilio"}
    else:
        user_data[from_number]["provider"] = "twilio"
    
    user_lang = user_data[from_number]["language"]
    
    # --- Language selection logic ---
    if user_lang == "unknown" or msg_body.lower() in ["/start", "hello", "hi", "habari"]:
        # Ask for language
        send_message(
            from_number,
            "Karibu / Welcome to Legal Bot! Please choose your language:\n"
            "Reply *1* for English\n"
            "Reply *2* for Kiswahili",
            provider="twilio"
        )
        return PlainTextResponse("OK", status_code=200)
    
    # Check if they are replying to language prompt
    if msg_body == "1":
        user_data[from_number]["language"] = "en"
        send_message(from_number, "Language set to English. Ask your legal question.", "twilio")
        return PlainTextResponse("OK")
    elif msg_body == "2":
        user_data[from_number]["language"] = "sw"
        send_message(from_number, "Lugha imewekwa Kiswahili. Uliza swali lako la kisheria.", "twilio")
        return PlainTextResponse("OK")
    
    # --- Process legal query (same as before) ---
    # same answer_conversational call, but we need to provide phone and lang
    try:
        result = answer_conversational(msg_body, user_lang, from_number)
        answer = result["answer"]
        disclaimer = ... 
        full_reply = answer + disclaimer  # same disclaimer logic
        send_message(from_number, full_reply, "twilio")
    except Exception as e:
        print("Error:", e)
        send_message(from_number, "Sorry, an error occurred.", "twilio")
    
    return PlainTextResponse("OK")