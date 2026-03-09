from __future__ import annotations
import logging
from typing import Optional
from mistralai import Mistral
from app.core.vector_store import search_similar_chunks
from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mistral client — singleton
# ---------------------------------------------------------------------------

_mistral_client: Optional[Mistral] = None


def get_mistral_client() -> Mistral:
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=settings.MISTRAL_API_KEY)
    return _mistral_client


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise document assistant. Your ONLY job is to answer questions based on the provided context from PDF documents.

Rules you must follow:
1. Answer ONLY from the provided context. Do NOT use outside knowledge.
2. If the answer is not in the context, respond: "I couldn't find information about this in the uploaded documents."
3. Be concise but complete.
4. When relevant, mention which part of the document your answer comes from.
5. If asked in Arabic, respond in Arabic. If in English, respond in English."""


def build_rag_prompt(context_chunks: list[dict], question: str) -> list[dict]:
    """
    Build the messages array for the Mistral API.

    RAG Augmentation pattern:
    - We inject retrieved chunks as context into the prompt
    - This "grounds" the LLM — it can only use what we give it
    - The context + question together form the augmented input
    """
    # Format context with source info
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Excerpt {i} from '{chunk['source']}' "
            f"(relevance: {chunk['similarity_score']:.0%})]:\n{chunk['text']}"
        )

    context_text = "\n\n" + "─" * 40 + "\n\n".join(context_parts)

    user_message = f"""Here is the relevant context from the documents:

{context_text}

─────────────────────────────────────
Question: {question}

Please answer based strictly on the context above."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


# ---------------------------------------------------------------------------
# Main RAG pipeline
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    source_filter: Optional[str] = None,
    top_k: Optional[int] = None,
) -> dict:
    """
    Full RAG pipeline:
    R → Retrieve relevant chunks from ChromaDB
    A → Augment the prompt with retrieved context
    G → Generate answer using Mistral

    Args:
        question: User's question
        source_filter: Limit search to a specific PDF filename
        top_k: Number of chunks to retrieve (default from settings)

    Returns:
        dict with answer, sources, and debug info
    """
    logger.info(f"RAG query: '{question[:80]}...' " if len(question) > 80 else f"RAG query: '{question}'")

    # ── Step 1: RETRIEVE ──────────────────────────────────────────────────
    relevant_chunks = search_similar_chunks(
        query=question,
        top_k=top_k or settings.TOP_K_RESULTS,
        source_filter=source_filter,
    )

    if not relevant_chunks:
        return {
            "answer": "No documents found in the system. Please upload a PDF first.",
            "sources": [],
            "chunks_used": 0,
            "model": settings.MISTRAL_MODEL,
            "top_chunks": [],
        }

    # Filter out low-relevance chunks (similarity < 0.3)
    filtered_chunks = [c for c in relevant_chunks if c["similarity_score"] >= 0.3]

    if not filtered_chunks:
        return {
            "answer": "I found some documents but none contained information relevant enough to answer your question. Try rephrasing or uploading more relevant documents.",
            "sources": [],
            "chunks_used": 0,
            "model": settings.MISTRAL_MODEL,
            "top_chunks": [],
        }

    logger.info(f"Retrieved {len(filtered_chunks)} relevant chunks (filtered from {len(relevant_chunks)})")

    # ── Step 2: AUGMENT ───────────────────────────────────────────────────
    messages = build_rag_prompt(filtered_chunks, question)

    # ── Step 3: GENERATE ──────────────────────────────────────────────────
    client = get_mistral_client()

    response = client.chat.complete(
        model=settings.MISTRAL_MODEL,
        messages=messages,
        temperature=0.1,      # Low temperature = more factual, less creative
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    # Collect unique sources
    sources_used = list({c["source"] for c in filtered_chunks})

    logger.info(f"Generated answer ({len(answer)} chars) from {len(filtered_chunks)} chunks")

    return {
        "answer": answer,
        "sources": sources_used,
        "chunks_used": len(filtered_chunks),
        "model": settings.MISTRAL_MODEL,
        "top_chunks": [
            {
                "text": c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
                "source": c["source"],
                "similarity": c["similarity_score"],
            }
            for c in filtered_chunks[:2]   # Show top 2 for transparency
        ],
    }
