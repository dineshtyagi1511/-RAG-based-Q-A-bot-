"""
ingest.py — Document ingestion pipeline.

Flow:
    File / Directory
        │
        ▼  (PyPDFLoader | TextLoader | DirectoryLoader)
    Raw Documents
        │
        ▼  (RecursiveCharacterTextSplitter)
    Chunks
        │
        ▼  (OpenAIEmbeddings)
    Vectors
        │
        ▼  (Qdrant HTTP / Cloud)
    Stored in collection QDRANT_COLLECTION

Also exposes `get_vectorstore()` for the retrieval side.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
)

logger = logging.getLogger(__name__)


# ── Qdrant client factory ────────────────────────────────────────────────────

from qdrant_client import QdrantClient

def _qdrant_client() -> QdrantClient:
    kwargs = {
        "url": QDRANT_URL,
        "timeout": 30,                 # Increase request timeout
        "check_compatibility": False,  # Avoid version check warning
    }

    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY

    return QdrantClient(**kwargs)


# ── Embeddings factory (reused across ingest & retrieval) ────────────────────

def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )


# ── Step 1: Load ─────────────────────────────────────────────────────────────

def load_documents(source: str) -> list[Document]:
    """Load PDF, TXT, MD, or a whole directory of mixed docs."""
    path = Path(source)
    docs: list[Document] = []

    if path.is_dir():
        for glob, loader_cls in [
            ("**/*.pdf", PyPDFLoader),
            ("**/*.txt", TextLoader),
            ("**/*.md", UnstructuredMarkdownLoader),
        ]:
            dl = DirectoryLoader(
                str(path),
                glob=glob,
                loader_cls=loader_cls,
                show_progress=False,
                silent_errors=True,
            )
            docs.extend(dl.load())
    elif source.lower().endswith(".pdf"):
        docs = PyPDFLoader(source).load()
    elif source.lower().endswith(".md"):
        docs = UnstructuredMarkdownLoader(source).load()
    else:
        docs = TextLoader(source, encoding="utf-8").load()

    logger.info("📄 Loaded %d document(s) from %s", len(docs), source)
    if not docs:
        raise ValueError(f"No documents found at: {source}")
    return docs


# ── Step 2: Chunk ─────────────────────────────────────────────────────────────

def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info("✂️  Created %d chunks (size=%d, overlap=%d)",
                len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


# ── Step 3 & 4: Embed + Store in Qdrant ──────────────────────────────────────

def _ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection only if it doesn't already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info("🗂️  Created Qdrant collection '%s'", QDRANT_COLLECTION)
    else:
        logger.info("🗂️  Using existing collection '%s'", QDRANT_COLLECTION)


def ingest_documents(source: str) -> dict[str, Any]:
    """
    Full ingest pipeline.  Returns a result dict for the Streamlit UI.

    Args:
        source: path to a PDF / TXT / MD file, or a directory.
    """
    try:
        raw  = load_documents(source)
        chunks = chunk_documents(raw)

        client = _qdrant_client()
        _ensure_collection(client)

        emb = _embeddings()
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=emb,
        )
        vectorstore.add_documents(chunks)

        result: dict[str, Any] = {
            "success": True,
            "source": source,
            "documents_loaded": len(raw),
            "chunks_created": len(chunks),
            "collection": QDRANT_COLLECTION,
        }
        logger.info("✅ Ingestion complete: %s", result)
        return result

    except Exception as exc:
        logger.exception("Ingestion failed")
        return {"success": False, "error": str(exc)}


# ── Retrieval helper ─────────────────────────────────────────────────────────

def get_vectorstore() -> QdrantVectorStore:
    """Return a QdrantVectorStore connected to the existing collection."""
    return QdrantVectorStore(
        client=_qdrant_client(),
        collection_name=QDRANT_COLLECTION,
        embedding=_embeddings(),
    )