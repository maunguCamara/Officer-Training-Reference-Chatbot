"""
services/whatsapp_session.py

Lightweight in-memory session store for WhatsApp users.
Tracks language preference and recent conversation history per phone number.

For production: replace _store with Redis or a simple DB table.
"""

from typing import Any

# In-memory store: { "whatsapp:+2547XXXXXXXX": { language, history } }
_store: dict[str, dict[str, Any]] = {}


def get_session(from_number: str) -> dict[str, Any]:
    """Return the session dict for a WhatsApp number, creating it if needed."""
    if from_number not in _store:
        _store[from_number] = {"language": "en", "history": []}
    return _store[from_number]


def update_session(from_number: str, updates: dict[str, Any]) -> None:
    """Merge updates into a user's session."""
    session = get_session(from_number)
    session.update(updates)
    _store[from_number] = session


def set_language(from_number: str, language: str) -> None:
    """Set the preferred language and reset conversation history."""
    _store[from_number] = {"language": language, "history": []}
