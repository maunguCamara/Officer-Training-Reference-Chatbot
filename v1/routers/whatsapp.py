"""
routers/whatsapp.py

Handles inbound WhatsApp messages via Twilio webhook.
Plugs directly into the existing RAG + Claude pipeline.

Flow:
  1. Twilio sends POST to /whatsapp/webhook when a user messages your number
  2. We look up (or create) the user's session & language preference
  3. We call the same RAG logic used by the mobile app
  4. We reply via Twilio's API
"""

from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse
from services.vector_store import query_chunks
from services.llm import ask_claude
from services.whatsapp_session import get_session, update_session, set_language
from services.twilio_client import send_whatsapp_message
from config import get_settings
import hmac, hashlib, urllib.parse

settings = get_settings()
router = APIRouter(prefix="/whatsapp", tags=["whatsapp"])

# ── Language command keywords ────────────────────────────────────────────────
LANG_COMMANDS = {
    "english": "en", "eng": "en", "en": "en",
    "swahili": "sw", "kiswahili": "sw", "sw": "sw",
}

HELP_TEXT = {
    "en": (
        "👋 *TrainingBot* — your training assistant\n\n"
        "Just ask any question about your training materials and I'll answer with sources.\n\n"
        "Commands:\n"
        "• Type *english* or *swahili* to switch language\n"
        "• Type *help* to see this message"
    ),
    "sw": (
        "👋 *TrainingBot* — msaidizi wako wa mafunzo\n\n"
        "Uliza swali lolote kuhusu nyaraka za mafunzo na nitajibu na vyanzo.\n\n"
        "Amri:\n"
        "• Andika *english* au *swahili* kubadilisha lugha\n"
        "• Andika *help* kuona ujumbe huu"
    ),
}

NO_ANSWER_TEXT = {
    "en": "I couldn't find an answer to that in the training materials. Try rephrasing your question.",
    "sw": "Sikupata jibu kwa hilo katika nyaraka za mafunzo. Jaribu kuuliza tena kwa njia tofauti.",
}


def verify_twilio_signature(request_url: str, post_body: bytes, signature: str) -> bool:
    """Validate that the webhook genuinely came from Twilio."""
    if not settings.twilio_auth_token:
        return True  # Skip in development
    params = dict(urllib.parse.parse_qsl(post_body.decode()))
    sorted_params = "".join(f"{k}{v}" for k, v in sorted(params.items()))
    expected = hmac.new(
        settings.twilio_auth_token.encode(),
        (request_url + sorted_params).encode(),
        hashlib.sha1,
    ).digest()
    import base64
    expected_b64 = base64.b64encode(expected).decode()
    return hmac.compare_digest(expected_b64, signature)


def format_citations(citations, language: str) -> str:
    """Format citations as readable WhatsApp text."""
    if not citations:
        return ""
    label = "📚 *Sources:*" if language == "en" else "📚 *Vyanzo:*"
    lines = [label]
    for c in citations:
        lines.append(f"  • _{c.title}_, page {c.page}")
    return "\n".join(lines)


@router.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    Twilio webhook endpoint.
    Twilio sends form-encoded POST data for each incoming WhatsApp message.
    """
    body = await request.body()

    # Optional: verify signature in production
    signature = request.headers.get("X-Twilio-Signature", "")
    if settings.twilio_auth_token:
        url = str(request.url)
        if not verify_twilio_signature(url, body, signature):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    # Parse form data
    params = dict(urllib.parse.parse_qsl(body.decode()))
    from_number = params.get("From", "")   # e.g. "whatsapp:+254712345678"
    message_text = params.get("Body", "").strip()

    if not from_number or not message_text:
        return PlainTextResponse("ok")

    # Load user session (language preference + conversation history)
    session = get_session(from_number)
    language = session.get("language", "en")

    # ── Handle commands ──────────────────────────────────────────────────────
    lower = message_text.lower().strip()

    if lower in ("help", "msaada"):
        send_whatsapp_message(from_number, HELP_TEXT[language])
        return PlainTextResponse("ok")

    if lower in LANG_COMMANDS:
        new_lang = LANG_COMMANDS[lower]
        set_language(from_number, new_lang)
        confirm = {
            "en": "✅ Language set to *English*. Ask your question!",
            "sw": "✅ Lugha imewekwa kuwa *Kiswahili*. Uliza swali lako!",
        }
        send_whatsapp_message(from_number, confirm[new_lang])
        return PlainTextResponse("ok")

    # ── RAG pipeline ─────────────────────────────────────────────────────────
    # Retrieve relevant chunks
    chunks = query_chunks(message_text, top_k=settings.top_k_chunks)

    if not chunks:
        send_whatsapp_message(from_number, NO_ANSWER_TEXT[language])
        return PlainTextResponse("ok")

    # Ask Claude (reusing the same service as the mobile app)
    history = session.get("history", [])
    chat_response = ask_claude(
        question=message_text,
        chunks=chunks,
        language=language,
        conversation_history=history[-6:],  # last 3 turns
    )

    # Build reply
    citations_text = format_citations(chat_response.citations, language)
    reply = chat_response.answer
    if citations_text:
        reply = f"{reply}\n\n{citations_text}"

    # Persist conversation turn
    history.append({"role": "user", "content": message_text})
    history.append({"role": "assistant", "content": chat_response.answer})
    update_session(from_number, {"history": history[-10:]})  # keep last 5 turns

    send_whatsapp_message(from_number, reply)
    return PlainTextResponse("ok")


@router.get("/webhook")
async def whatsapp_verify(request: Request):
    """
    Meta/Twilio webhook verification (GET challenge).
    Only needed if using Meta's WhatsApp Business API directly.
    """
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == settings.whatsapp_verify_token:
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")
