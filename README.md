# 🔍 RAG Q&A Bot

> **Production-grade Retrieval-Augmented Generation** chatbot built with
> LangChain · Qdrant · Redis · LLM Guard · Streamlit.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Streamlit UI                          │
│            (chat interface + sidebar config)                 │
└─────────────────────────┬────────────────────────────────────┘
                          │ user query
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                  SECURITY LAYER (security.py)                │
│                                                              │
│  Tier 1 ── Input Sanitisation  (regex, always active)        │
│  Tier 2 ── Rate Limiting       (Redis sliding-window)        │
│  Tier 3 ── LLM Guard Scanners  (PromptInjection, Toxicity)   │
└─────────────────────────┬────────────────────────────────────┘
                          │ clean query
                          ▼
            ┌─────────────────────────┐
            │      Redis Cache        │◄─── cache HIT → return
            │  SHA-256 keyed, TTL 1h  │
            └────────────┬────────────┘
                         │ cache MISS
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                 RAG PIPELINE (rag_pipeline.py)               │
│                                                              │
│   Qdrant Retriever                                           │
│   (cosine similarity, HNSW, top-k chunks)                   │
│          │                                                   │
│          ▼                                                   │
│   LangChain LCEL Chain                                       │
│   (PromptTemplate | ChatOpenAI | StrOutputParser)           │
│          │                                                   │
│          ▼                                                   │
│   LLM Guard Output Scan  (Relevance scanner)                 │
└─────────────────────────┬────────────────────────────────────┘
                          │ answer + sources
                          ▼
            ┌─────────────────────────┐
            │  Redis Cache SET        │
            └─────────────────────────┘
                          │
                          ▼
                  Chat UI response
```

### Ingestion Pipeline (offline)

```
File / Directory
      │
      ▼  (PyPDF / TextLoader / DirectoryLoader)
  Raw Documents
      │
      ▼  (RecursiveCharacterTextSplitter, 500 chars / 50 overlap)
  Chunks
      │
      ▼  (OpenAI text-embedding-3-small → 1536-dim vectors)
  Embeddings
      │
      ▼  (QdrantVectorStore.add_documents)
  Qdrant Collection
```

---

## Features

| Feature | Detail |
|---|---|
| **LLM** | GPT-4o-mini (configurable) |
| **Embeddings** | `text-embedding-3-small` (1536 dims) |
| **Vector DB** | Qdrant — HNSW index, cosine similarity |
| **Cache** | Redis — SHA-256 keyed, configurable TTL |
| **Security Tier 1** | Regex-based input sanitisation & injection detection |
| **Security Tier 2** | Redis sliding-window rate limiter |
| **Security Tier 3** | LLM Guard (PromptInjection + Toxicity + Relevance) |
| **Deduplication** | Automatic point ID verification before upload to skip duplicates |
| **Document types** | PDF, TXT, Markdown, mixed directories |
| **UI** | Glassmorphic cyber-dark UI, high-contrast typography, styled citation cards, cache metrics |
| **Deployment** | Render, Streamlit Cloud ready |

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd rag_qa_bot

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root and set the required variables
(see [Configuration Reference](#configuration-reference)) — at minimum
`OPENAI_API_KEY`, plus your Qdrant Cloud and Upstash Redis credentials.
See [Using Cloud Services](#using-cloud-services) below for setup steps —
both have generous free tiers, so no local database containers are needed.

### 3. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501, enter your OpenAI API key in the sidebar,
then upload a document to start chatting.

---

## Using Cloud Services

### Qdrant Cloud (free tier)
1. Sign up at https://cloud.qdrant.io
2. Create a cluster → copy the URL and API key
3. Update `.env`:
   ```
   QDRANT_URL=https://your-cluster.qdrant.io
   QDRANT_API_KEY=your-qdrant-api-key
   ```

### Upstash Redis (free tier, serverless)
1. Sign up at https://upstash.com
2. Create a Redis database → copy the REST URL
3. Update `.env`:
   ```
   REDIS_URL=rediss://default:your-token@your-endpoint.upstash.io:6379
   ```

### Render Deployment
1. Push your project to a GitHub repository.
2. Go to [Render](https://render.com) → **New** → **Web Service** → connect your repo.
3. Set the **Build Command**: `pip install -r requirements.txt`
4. Set the **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Under **Environment**, add your variables (`OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `REDIS_URL`, etc.)
6. Click **Create Web Service** to deploy.

### Streamlit Cloud Deployment
1. Push your project files to a GitHub repository.
2. Go to [Streamlit Share](https://share.streamlit.io/) and click **"New App"**.
3. Select your repository, set the branch to `main`, and main file path to `app.py`.
4. Click **"Advanced settings..."** and input your environment secrets in the **Secrets** text area (in TOML format):
   ```toml
   OPENAI_API_KEY = "sk-..."
   QDRANT_URL = "https://..."
   QDRANT_API_KEY = "..."
   QDRANT_COLLECTION = "documents"
   REDIS_URL = "rediss://..."
   ```
5. Click **"Deploy!"**.

---

## Enabling LLM Guard (Tier 3 Security)

LLM Guard requires PyTorch. Install CPU-only torch first to keep the image lean:

```bash
# CPU-only (recommended for cloud/container deployments)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install llm-guard

# Then enable in .env
ENABLE_LLM_GUARD=true
```

Scanners activated:
- **Input** → `TokenLimit` · `PromptInjection` · `Toxicity`
- **Output** → `Relevance`

Without LLM Guard, Tiers 1 & 2 remain fully active.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (required) |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBEDDING_DIM` | `1536` | Vector dimensions |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_API_KEY` | — | Qdrant Cloud key |
| `QDRANT_COLLECTION` | `rag_knowledge_base` | Collection name |
| `REDIS_URL` | `redis://localhost:6379` | Redis endpoint |
| `REDIS_TTL` | `3600` | Cache TTL (seconds) |
| `RATE_LIMIT_REQUESTS` | `10` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Window size (seconds) |
| `MAX_QUERY_LENGTH` | `500` | Max query characters |
| `ENABLE_LLM_GUARD` | `false` | Enable ML-based scanning |
| `CHUNK_SIZE` | `500` | Document chunk size |
| `CHUNK_OVERLAP` | `50` | Chunk overlap |
| `TOP_K` | `4` | Chunks retrieved per query |

---

## Project Structure

```
rag_qa_bot/
├── .gitignore
├── LICENSE
├── README.md
├── app.py                 ← Streamlit UI (entry point)
├── cache.py               ← Redis response cache (get/set/invalidate/stats)
├── config.py              ← Centralised env configuration
├── embedding_cache.py     ← Redis-backed embedding cache (skips re-embedding)
├── embedding_utils.py     ← Embedding generation helpers
├── hashing.py             ← SHA-256 hashing utilities (cache keys, dedup)
├── ingest.py              ← Document ingestion → Qdrant
├── qdrant_utils.py        ← Qdrant client & collection helper functions
├── rag_pipeline.py        ← LCEL RAG chain + cache integration
├── requirements.txt
├── security.py            ← 3-tier security layer
├── docs/
│   └── sample_ai_tutorial.txt   ← Demo knowledge base
└── temp_uploads/          ← Scratch space for files awaiting ingestion
```

---

## Sample Questions (using the bundled tutorial)

- *"What is the difference between supervised and unsupervised learning?"*
- *"How does RAG reduce hallucinations?"*
- *"What embedding models does OpenAI offer and what are their dimensions?"*
- *"What is HNSW and which vector databases use it?"*
- *"What are the OWASP Top 10 for LLM applications?"*
- *"How does LLM Guard protect against prompt injection?"*
- *"What is the role of Redis in production AI systems?"*

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | [LangChain](https://python.langchain.com) 0.3 (LCEL) |
| LLM | [OpenAI](https://platform.openai.com) GPT-4o-mini |
| Vector DB | [Qdrant](https://qdrant.tech) (Cloud) |
| Cache | [Redis](https://redis.io) (Upstash) |
| Security | [LLM Guard](https://llm-guard.com) (optional) |
| UI | [Streamlit](https://streamlit.io) |
| Deployment | [Render](https://render.com) |