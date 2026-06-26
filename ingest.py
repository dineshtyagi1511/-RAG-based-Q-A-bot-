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
from datetime import datetime, UTC


from hashing import sha256_text

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


from qdrant_utils import get_qdrant_client
from embedding_utils import get_embeddings



# ── Embeddings factory (reused across ingest & retrieval) ────────────────────

def get_embeddings() -> OpenAIEmbeddings:
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
        try:
            docs = TextLoader(source, encoding="utf-8").load()
        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed for %s, retrying with latin-1", source)
            docs = TextLoader(source, encoding="latin-1").load()

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
    
    # Assign metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_hash"] = sha256_text(chunk.page_content)

    logger.info("✂️  Created %d chunks (size=%d, overlap=%d) with metadata",
                len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    
    return chunks


# ── Step 3 & 4: Embed + Store in Qdrant ──────────────────────────────────────

from qdrant_client.models import Distance, VectorParams

def _ensure_collection(client: QdrantClient) -> None:
    """
    Create the collection only if it doesn't exist.
    Safe to call multiple times.
    """

    if client.collection_exists(QDRANT_COLLECTION):
        logger.info(
            "🗂️ Using existing collection '%s'",
            QDRANT_COLLECTION,
        )
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )

    logger.info(
        "🗂️ Created collection '%s'",
        QDRANT_COLLECTION,
    )


from qdrant_client.http import models
from embedding_cache import EmbeddingCache
import uuid

def ingest_documents(source: str) -> dict[str, Any]:
    try:
        # 1. Load and Hash Documents
        raw = load_documents(source)
        for doc in raw:
            doc.metadata["document_hash"] = sha256_text(doc.page_content)
        
        # 2. Chunking (with metadata enrichment as per Step 4)
        chunks = chunk_documents(raw)
        if not chunks:
            raise ValueError("No text chunks could be extracted from the document.")

        # 2b. Deduplication: Check if the document has already been ingested using first chunk's deterministic point ID
        client = get_qdrant_client()
        if client.collection_exists(QDRANT_COLLECTION):
            first_chunk_hash = chunks[0].metadata["chunk_hash"]
            first_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, first_chunk_hash))
            existing_points = client.retrieve(
                collection_name=QDRANT_COLLECTION,
                ids=[first_point_id],
            )
            if existing_points:
                result = {
                    "success": True,
                    "already_exists": True,
                    "source": source,
                    "documents_loaded": len(raw),
                    "chunks_created": len(chunks),
                    "collection": QDRANT_COLLECTION,
                }
                logger.info("ℹ️ Document already ingested (skipping upload): %s", result)
                return result
        
        # 3. Generate/Fetch Embeddings with Cache
        emb = get_embeddings()
        vectors = []
        for chunk in chunks:
            vector = EmbeddingCache.get(chunk.page_content)
            if vector is None:
                vector = emb.embed_query(chunk.page_content)
                EmbeddingCache.set(chunk.page_content, vector)
            vectors.append(vector)
            
        # 4. Build PointStructs (Deterministic IDs)
        points = []
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.metadata["chunk_hash"]))
            
            # Enrich metadata
            chunk.metadata.update({
                "source": source,
                "embedding_model": EMBEDDING_MODEL,
                "ingested_at": datetime.utcnow().isoformat(),
            })
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vectors[i],
                    payload={
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata,
                    }
                )
            )
            
        # 5. Batch Upsert
        client = get_qdrant_client()
        _ensure_collection(client)
        
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points[i:i + BATCH_SIZE],
                wait=True,
            )

        result = {
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
        client=get_qdrant_client(),
        collection_name=QDRANT_COLLECTION,
        embedding=get_embeddings(),
    )