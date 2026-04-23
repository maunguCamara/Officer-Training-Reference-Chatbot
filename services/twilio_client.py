"""
services/twilio_client.py

Sends WhatsApp messages via Twilio's REST API.
No SDK needed — plain HTTPS requests.
"""

import httpx
from config import get_settings

settings = get_settings()

TWILIO_API_URL = (
    f"https://api.twilio.com/2010-04-01/Accounts/"
    f"{settings.twilio_account_sid}/Messages.json"
)


def send_whatsapp_message(to: str, body: str) -> None:
    """
    Send a WhatsApp message via Twilio.

    Args:
        to:   Recipient in Twilio format, e.g. "whatsapp:+254712345678"
        body: Message text (supports WhatsApp markdown: *bold*, _italic_)
    """
    # Truncate to WhatsApp's 4096-char limit
    if len(body) > 4000:
        body = body[:3997] + "…"

    with httpx.Client() as client:
        response = client.post(
            TWILIO_API_URL,
            data={
                "From": f"whatsapp:{settings.twilio_whatsapp_number}",
                "To": to,
                "Body": body,
            },
            auth=(settings.twilio_account_sid, settings.twilio_auth_token),
            timeout=10,
        )
        response.raise_for_status()
