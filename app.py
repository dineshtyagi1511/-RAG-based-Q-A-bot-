# NOTE: Placeholder copy of uploaded app.py.
# Due to response/tool limits, automated refactor wasn't applied in this file.
# Apply the config changes discussed (from config import settings, replace os.getenv usage, etc.).

"""
app.py — Streamlit UI for the RAG Q&A Bot.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import streamlit as st

# ── Page config must be the FIRST Streamlit call ─────────────────────────────
st.set_page_config(
    page_title="RAG Q&A Bot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }

/* ── header ── */
.rag-header {
    background: linear-gradient(135deg, #1f2d3d 0%, #0d1117 60%, #1a1f35 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    text-align: center;
}
.rag-header h1 { color: #e6edf3; font-size: 2rem; margin: 0; }
.rag-header p  { color: #8b949e; margin: 6px 0 0 0; }

/* ── tech pills ── */
.pill-row { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-bottom: 20px; }
.pill {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    color: #8b949e;
    font-weight: 500;
}

/* ── message metadata row ── */
.meta-row {
    display: flex; gap: 12px; flex-wrap: wrap;
    margin-top: 8px;
    font-size: 11px;
    color: #8b949e;
}
.meta-badge {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 2px 8px;
}
.meta-badge.hit   { border-color: #238636; color: #3fb950; }
.meta-badge.fresh { border-color: #1f6feb; color: #58a6ff; }
.meta-badge.warn  { border-color: #9e6a03; color: #d29922; }

/* ── sidebar labels ── */
.sidebar-section {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b949e;
    margin: 16px 0 6px 0;
    font-weight: 600;
}

/* ── status indicator ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.dot-green  { background: #3fb950; }
.dot-yellow { background: #d29922; }
.dot-red    { background: #f85149; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "session_id": str(uuid.uuid4())[:8],
        "chat_history": [],        # list of {role, content, meta?}
        "ingested": False,
        "doc_label": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.markdown("### 🔍 RAG Q&A Bot")
        st.caption(f"Session `{st.session_state.session_id}`")

        # ── API key ──────────────────────────────────────────────────────────
        from config import settings

        st.markdown('<p class="sidebar-section">🔑 OpenAI</p>', unsafe_allow_html=True)

        key_ok = bool(settings.OPENAI_API_KEY)

        st.markdown(
            f'<span class="status-dot {"dot-green" if key_ok else "dot-red"}"></span>'
            f'{"Loaded from .env" if key_ok else "Missing in .env"}',
            unsafe_allow_html=True,
        )

        # ── Document ingestion ────────────────────────────────────────────────
        st.markdown('<p class="sidebar-section">📚 Knowledge Base</p>', unsafe_allow_html=True)

        tab_up, tab_path = st.tabs(["Upload", "Local Path"])

        with tab_up:
            uploaded = st.file_uploader(
                "PDF / TXT / MD",
                type=["pdf", "txt", "md"],
                label_visibility="collapsed",
            )
            if uploaded and st.button("⚡ Ingest", key="btn_upload"):
                if not key_ok:
                    st.error("Set OpenAI API key first.")
                else:
                    save_path = f"/tmp/{uploaded.name}"
                    with open(save_path, "wb") as fh:
                        fh.write(uploaded.getbuffer())
                    _do_ingest(save_path, uploaded.name)

        with tab_path:
            doc_path = st.text_input(
                "File / directory path",
                value="docs/sample_ai_tutorial.txt",
                label_visibility="collapsed",
            )
            if st.button("⚡ Ingest", key="btn_path"):
                if not key_ok:
                    st.error("Set OpenAI API key first.")
                elif not os.path.exists(doc_path):
                    st.error(f"Path not found: `{doc_path}`")
                else:
                    _do_ingest(doc_path, Path(doc_path).name)

        if st.session_state.ingested:
            st.success(f"✅ **{st.session_state.doc_label}** loaded")

        st.divider()

        # ── Cache stats ───────────────────────────────────────────────────────
        st.markdown('<p class="sidebar-section">⚡ Redis Cache</p>', unsafe_allow_html=True)
        try:
            from cache import cache
            stats = cache.stats()
            if stats["status"] == "connected":
                col1, col2 = st.columns(2)
                col1.metric("Cached", stats["cached_queries"])
                col2.metric("TTL", f'{stats["ttl_seconds"]}s')
                st.caption(f"Hits: {stats['hits']}  |  Misses: {stats['misses']}")
                if st.button("🗑️ Clear cache"):
                    n = cache.invalidate_all()
                    st.toast(f"Cleared {n} entries", icon="🗑️")
                    st.rerun()
            else:
                st.warning(f"⚠️ {stats['status']}")
        except Exception as exc:
            st.warning(f"Cache error: {exc}")

        st.divider()

        # ── Security status ───────────────────────────────────────────────────
        st.markdown('<p class="sidebar-section">🛡️ Security</p>', unsafe_allow_html=True)
        from config import ENABLE_LLM_GUARD, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW

        rows = [
            ("Input sanitisation",  "✅ Active",  "dot-green"),
            (f"Rate limit",  f"{RATE_LIMIT_REQUESTS} req / {RATE_LIMIT_WINDOW}s", "dot-green"),
            ("LLM Guard", ("✅ Active" if ENABLE_LLM_GUARD else "⚠️ Disabled"),
             "dot-green" if ENABLE_LLM_GUARD else "dot-yellow"),
        ]
        for label, val, dot in rows:
            st.markdown(
                f'<span class="status-dot {dot}"></span>**{label}:** {val}',
                unsafe_allow_html=True,
            )


def _do_ingest(path: str, label: str):
    """Run ingestion with a spinner and update session state."""
    from ingest import ingest_documents

    with st.spinner(f"Ingesting **{label}**…"):
        result = ingest_documents(path)

    if result["success"]:
        st.success(
            f"✅ {result['documents_loaded']} doc(s) → "
            f"{result['chunks_created']} chunks → Qdrant"
        )
        st.session_state.ingested = True
        st.session_state.doc_label = label
    else:
        st.error(f"❌ Ingestion failed: {result['error']}")


# ── Main area ─────────────────────────────────────────────────────────────────

def _main():
    # Header
    st.markdown("""
    <div class="rag-header">
        <h1>🔍 RAG Q&amp;A Bot</h1>
        <p>Ask questions — answers grounded in your documents</p>
    </div>
    <div class="pill-row">
        <span class="pill">🦜 LangChain</span>
        <span class="pill">🔷 Qdrant</span>
        <span class="pill">⚡ Redis</span>
        <span class="pill">🛡️ LLM Guard</span>
        <span class="pill">🤖 GPT-4o-mini</span>
    </div>
    """, unsafe_allow_html=True)

    # Onboarding hint
    if not st.session_state.ingested:
        st.info(
            "👈 **Get started:** Enter your OpenAI API key in the sidebar, "
            "then upload or point to a document to build the knowledge base."
        )

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                _render_meta(msg["meta"])

    # Input box
    query = st.chat_input("Ask a question about your documents…")
    if not query:
        return

    # ── Guard rails before hitting the pipeline ───────────────────────────────
    
    

    from security import run_input_security

    allowed, clean_query, err, remaining = run_input_security(
        query, st.session_state.session_id
    )
    if not allowed:
        st.error(err)
        return

    # Show user bubble
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run RAG
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base…"):
            from rag_pipeline import query_rag
            result = query_rag(clean_query)

        st.markdown(result["answer"])
        _render_meta(result, remaining=remaining)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": result["answer"], "meta": result}
    )


def _render_meta(meta: dict, remaining: int | None = None):
    """Render the metadata row below an assistant message."""
    cache_hit = meta.get("cache_hit", False)
    badge_cls = "hit" if cache_hit else "fresh"
    badge_txt = "🟢 Cache Hit" if cache_hit else "🔵 Fresh"

    cols = st.columns([2, 2, 2, 2])
    cols[0].caption(badge_txt)
    cols[1].caption(f"📦 {meta.get('chunks_retrieved', 0)} chunks")
    cols[2].caption(f"🤖 {meta.get('model', '—')}")
    if remaining is not None:
        cols[3].caption(f"🔒 {remaining} req left")

    if meta.get("sources"):
        with st.expander("📖 Sources"):
            for src in meta["sources"]:
                st.markdown(f"**📄 {src['filename']}** — page {src['page']}")
                st.caption(f"> {src['snippet']}")

    if meta.get("error"):
        st.caption(f"⚠️ Error detail: {meta['error']}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    _sidebar()
    _main()


if __name__ == "__main__":
    main()