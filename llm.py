"""
Claude API wrapper.
Builds RAG prompts with retrieved chunks and returns cited answers
in English or Swahili.
"""
import anthropic
from models import Citation, ChatResponse
from config import get_settings

settings = get_settings()
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT_EN = """You are a helpful training assistant for an organization. \
You answer questions based ONLY on the provided reference material (context chunks). \
Be clear, concise, and professional.

Rules:
- Answer ONLY from the provided context. If the answer is not in the context, say so honestly.
- Always cite your sources using [Doc: <title>, Page <number>] inline.
- If multiple chunks support your answer, cite all of them.
- Do not make up information.
- Keep answers focused and practical.
"""

SYSTEM_PROMPT_SW = """Wewe ni msaidizi wa mafunzo kwa shirika. \
Unajibu maswali kulingana NA nyaraka zilizotolewa tu (vipande vya muktadha). \
Kuwa wazi, mfupi, na wa kitaalamu.

Sheria:
- Jibu KUTOKA kwa muktadha uliotolewa tu. Kama jibu halikopo, sema ukweli.
- Daima taja vyanzo vyako kwa kutumia [Hati: <kichwa>, Ukurasa <namba>] ndani ya jibu.
- Ikiwa vipande vingi vinasaidia jibu lako, taja vyote.
- Usibuni taarifa.
- Weka majibu kuwa ya msingi na ya vitendo.
"""


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    lines = ["=== REFERENCE MATERIAL ===\n"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[{i}] Source: {chunk['title']} | Page {chunk['page']}\n"
            f"{chunk['text']}\n"
        )
    return "\n".join(lines)


def ask_claude(
    question: str,
    chunks: list[dict],
    language: str = "en",
    conversation_history: list[dict] | None = None,
) -> ChatResponse:
    """
    Send question + retrieved chunks to Claude and parse the response.
    Returns ChatResponse with answer text and structured citations.
    """
    system = SYSTEM_PROMPT_SW if language == "sw" else SYSTEM_PROMPT_EN
    context = build_context_block(chunks)

    # Build message history
    messages = list(conversation_history or [])
    messages.append(
        {
            "role": "user",
            "content": f"{context}\n\n---\nQuestion: {question}",
        }
    )

    response = _client.messages.create(
        model=settings.claude_model,
        max_tokens=settings.claude_max_tokens,
        system=system,
        messages=messages,
    )

    answer_text = response.content[0].text

    # Build citations from the chunks that were retrieved
    # (Claude cites them inline; we surface all retrieved sources)
    citations = []
    seen = set()
    for chunk in chunks:
        key = (chunk["document_id"], chunk["page"])
        if key not in seen:
            seen.add(key)
            # Only include if chunk is plausibly referenced (score threshold)
            if chunk.get("score", 1) >= 0.3:
                citations.append(
                    Citation(
                        document_id=chunk["document_id"],
                        title=chunk["title"],
                        page=chunk["page"],
                        excerpt=chunk["text"][:200] + "…"
                        if len(chunk["text"]) > 200
                        else chunk["text"],
                    )
                )

    return ChatResponse(
        answer=answer_text,
        citations=citations,
        language=language,
    )
