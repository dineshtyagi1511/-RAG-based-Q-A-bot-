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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── global theme overrides ── */
[data-testid="stAppViewContainer"] {
    background: #090d16 !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0 !important;
}
[data-testid="stHeader"] {
    background: rgba(9, 13, 22, 0.6) !important;
    backdrop-filter: blur(12px);
}
[data-testid="stSidebar"] {
    background: #0d121f !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.3) !important;
}

/* ── text visibility & color overrides ── */
/* Force all default paragraphs, list items, markdown texts, labels to be light grey / white */
[data-testid="stAppViewContainer"] p, 
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] ul,
[data-testid="stAppViewContainer"] ol,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span:not(.badge):not(.status-dot):not(.source-num):not(.pill) {
    color: #e2e8f0 !important;
}

/* Sidebar texts override */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span:not(.status-dot):not(.pill) {
    color: #cbd5e1 !important;
}

/* Typography overrides for headers */
h1, h2, h3, h4, h5, h6,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Captions and small stats */
[data-testid="stCaptionContainer"],
[data-testid="stAppViewContainer"] .stCaption, 
caption, 
.caption,
.meta-item,
.meta-item b {
    color: #94a3b8 !important;
}

/* Streamlit Tabs */
button[data-baseweb="tab"] {
    color: #94a3b8 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
    border-bottom-color: #8b5cf6 !important;
}
button[data-baseweb="tab"] p {
    color: inherit !important;
}

/* Form input, text inputs & textareas */
input, textarea, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
    color: #ffffff !important;
    background-color: #111827 !important;
}
div[data-baseweb="input"] {
    background-color: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
}
::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: rgba(17, 24, 39, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] details summary {
    color: #ffffff !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] details summary p {
    color: #ffffff !important;
}

/* ── scrollbars ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #090d16;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.2);
}

/* ── header ── */
.rag-header {
    background: linear-gradient(135deg, rgba(31, 41, 55, 0.3) 0%, rgba(17, 24, 39, 0.5) 100%);
    border: 1px solid rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 32px 24px;
    margin-bottom: 28px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}
.rag-header::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
}
.header-glow {
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.06) 0%, transparent 60%);
    pointer-events: none;
}
.rag-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    color: #f8fafc !important;
    font-size: 2.2rem;
    margin: 0;
    letter-spacing: -0.02em;
}
.rag-header p {
    color: #94a3b8 !important;
    margin: 8px 0 0 0;
    font-size: 14px;
}

/* ── tech pills ── */
.pill-row {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 24px;
}
.pill {
    background: rgba(31, 41, 55, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 30px;
    padding: 6px 16px;
    font-size: 12px;
    color: #94a3b8 !important;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}
.pill:hover {
    transform: translateY(-2px);
    border-color: rgba(139, 92, 246, 0.4);
    color: #c084fc !important;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2);
}

/* ── sidebar labels ── */
.sidebar-section {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b !important;
    margin: 24px 0 10px 0;
    font-weight: 700;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding-bottom: 6px;
}

/* ── status indicator ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 8px currentColor;
}
.dot-green  { background: #10b981; color: rgba(16, 185, 129, 0.4); }
.dot-yellow { background: #f59e0b; color: rgba(245, 158, 11, 0.4); }
.dot-red    { background: #ef4444; color: rgba(239, 68, 68, 0.4); }

/* ── message metadata & badges ── */
.meta-container {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    margin-top: 10px;
    padding: 8px 12px;
    background: rgba(31, 41, 55, 0.25);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.04);
}
.badge {
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}
.badge-hit {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399 !important;
    border: 1px solid rgba(16, 185, 129, 0.2);
}
.badge-fresh {
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa !important;
    border: 1px solid rgba(59, 130, 246, 0.2);
}
.meta-item {
    font-size: 12px;
    color: #94a3b8 !important;
}

/* ── sources display ── */
.source-card {
    background: rgba(15, 23, 42, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
    transition: all 0.2s ease;
}
.source-card:hover {
    border-color: rgba(139, 92, 246, 0.25);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
.source-card-header {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 6px;
    font-size: 12px;
}
.source-num {
    background: #8b5cf6;
    color: white !important;
    font-weight: bold;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 10px;
}
.source-file {
    font-weight: 600;
    color: #cbd5e1 !important;
}
.source-page {
    color: #64748b !important;
    margin-left: auto;
}
.source-snippet {
    font-size: 13px;
    color: #94a3b8 !important;
    line-height: 1.5;
    padding-left: 8px;
    border-left: 2px solid rgba(139, 92, 246, 0.3);
}

/* ── streamlit controls adjustments ── */
.stButton>button {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2) !important;
}
.stButton>button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4) !important;
    border: none !important;
}
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
                    temp_dir = Path("temp_uploads")
                    temp_dir.mkdir(exist_ok=True)
                    save_path = str(temp_dir / uploaded.name)
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
        rows = [
            ("Input sanitisation",  "✅ Active",  "dot-green"),
            (f"Rate limit",  f"{settings.RATE_LIMIT_REQUESTS} req / {settings.RATE_LIMIT_WINDOW}s", "dot-green"),
            ("LLM Guard", ("✅ Active" if settings.ENABLE_LLM_GUARD else "⚠️ Disabled"),
             "dot-green" if settings.ENABLE_LLM_GUARD else "dot-yellow"),
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
        if result.get("already_exists"):
            st.info(f"ℹ️ **{label}** has already been uploaded & ingested!")
        else:
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
        <div class="header-glow"></div>
        <h1>🧠 Neural RAG Q&amp;A Bot</h1>
        <p>Advanced Retrieval-Augmented Generation with real-time semantic caching</p>
    </div>
    <div class="pill-row">
        <span class="pill">🦜 LangChain</span>
        <span class="pill">🔷 Qdrant Cloud</span>
        <span class="pill">⚡ Redis Cache</span>
        <span class="pill">🛡️ Triple-Tier Security</span>
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
    badge_txt = (
        '<span class="badge badge-hit">🟢 Cache Hit</span>'
        if cache_hit else
        '<span class="badge badge-fresh">⚡ Fresh Gen</span>'
    )
    
    chunks = meta.get("chunks_retrieved", 0)
    model = meta.get("model", "—")
    
    meta_html = f"""
    <div class="meta-container">
        {badge_txt}
        <span class="meta-item">📦 <b>{chunks}</b> chunks</span>
        <span class="meta-item">🤖 <b>{model}</b></span>
    """
    if remaining is not None:
        meta_html += f'<span class="meta-item">🔒 <b>{remaining}</b> req remaining</span>'
    meta_html += "</div>"
    
    st.markdown(meta_html, unsafe_allow_html=True)

    if meta.get("sources"):
        with st.expander("📖 Grounded Sources"):
            for i, src in enumerate(meta["sources"], 1):
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-card-header">
                        <span class="source-num">#{i}</span>
                        <span class="source-file">📄 {src['filename']}</span>
                        <span class="source-page">page {src['page']}</span>
                    </div>
                    <div class="source-snippet">
                        {src['snippet']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    if meta.get("error"):
        st.caption(f"⚠️ Error detail: {meta['error']}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    _sidebar()
    _main()


if __name__ == "__main__":
    main()