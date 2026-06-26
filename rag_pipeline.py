"""
rag_pipeline.py — Core Retrieval-Augmented Generation pipeline.

Query flow:
    query
      │
      ├─── Redis cache HIT ──────────────────────────────► cached result
      │
      └─── cache MISS
              │
              ▼
          Qdrant retriever  (top-k semantic search)
              │
              ▼
          Prompt template  (context + question)
              │
              ▼
          OpenAI LLM  (gpt-4o-mini by default)
              │
              ▼
          LLM Guard output scan  (optional)
              │
              ▼
          Cache SET  →  return result dict
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from cache import cache
from config import LLM_MODEL, OPENAI_API_KEY, QDRANT_COLLECTION, TOP_K
from ingest import get_vectorstore
from security import scan_output

logger = logging.getLogger(__name__)

# ── Prompt ───────────────────────────────────────────────────────────────────

_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable AI assistant. Answer the user's question
using ONLY the information in the context blocks below.

Rules:
1. If the answer is not present in the context, say exactly:
   "I don't have enough information in the provided documents to answer that."
2. Never fabricate facts, names, or figures.
3. Cite the source filename when relevant.
4. Be concise but complete.

───────────────────────────────────────────────────
CONTEXT:
{context}
───────────────────────────────────────────────────

QUESTION: {question}

ANSWER:""",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        src  = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "—")
        parts.append(f"[{i}] Source: {src} | Page: {page}\n{doc.page_content}")
    return "\n\n".join(parts)


def _extract_sources(docs) -> list[dict[str, str]]:
    seen: set[str] = set()
    sources = []
    for doc in docs:
        src  = doc.metadata.get("source", "Unknown")
        page = str(doc.metadata.get("page", "N/A"))
        key  = f"{src}|{page}"
        if key not in seen:
            sources.append({
                "source": src,
                "filename": Path(src).name,
                "page": page,
                "snippet": doc.page_content[:250].strip() + "…",
            })
            seen.add(key)
    return sources


# ── Chain builder ─────────────────────────────────────────────────────────────

def _build_chain(retriever):
    """Build an LCEL chain: retriever → prompt → LLM → parser."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY,
    )
    return (
        {
            "context":  retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | _PROMPT
        | llm
        | StrOutputParser()
    )


# ── Public API ────────────────────────────────────────────────────────────────

def query_rag(query: str) -> dict[str, Any]:
    """
    Run a RAG query end-to-end.

    Returns a dict with keys:
        answer           str
        sources          list[dict]
        cache_hit        bool
        model            str
        chunks_retrieved int
        error            str  (only on failure)
    """
    # ── 1. Cache check ───────────────────────────────────────────────────────
    cached = cache.get(query, QDRANT_COLLECTION)
    if cached:
        cached["cache_hit"] = True
        return cached

    # ── 2. Retrieval ─────────────────────────────────────────────────────────
    try:
        vectorstore = get_vectorstore()
    except Exception as exc:
        logger.error("Qdrant connection failed: %s", exc)
        return {
            "answer": (
                "❌ Could not connect to the knowledge base. "
                "Please ensure Qdrant is running and documents have been ingested."
            ),
            "sources": [],
            "cache_hit": False,
            "error": str(exc),
            "model": LLM_MODEL,
            "chunks_retrieved": 0,
        }

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # Fetch source docs for citation (separate invoke to keep chain pure)
    source_docs = retriever.invoke(query)

    # ── 3. Generation ────────────────────────────────────────────────────────
    try:
        chain  = _build_chain(retriever)
        answer = chain.invoke(query)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        return {
            "answer": f"❌ Generation error: {exc}",
            "sources": _extract_sources(source_docs),
            "cache_hit": False,
            "error": str(exc),
            "model": LLM_MODEL,
            "chunks_retrieved": len(source_docs),
        }

    # ── 4. Output security scan (Tier 3) ─────────────────────────────────────
    is_safe, safe_answer, reason = scan_output(query, answer)
    if not is_safe:
        answer = (
            f"⚠️ This response was blocked by the output security policy.\n"
            f"Reason: {reason}"
        )
    else:
        answer = safe_answer

    # ── 5. Build result & cache ───────────────────────────────────────────────
    result: dict[str, Any] = {
        "answer": answer,
        "sources": _extract_sources(source_docs),
        "cache_hit": False,
        "model": LLM_MODEL,
        "chunks_retrieved": len(source_docs),
    }
    cache.set(query, QDRANT_COLLECTION, result)

    return result