"""
Vector retrieval layer.

Uses ChromaDB with its default sentence-transformer embedding function.
Every document is stored as a collection of overlapping text chunks; retrieval
returns the top-k chunks ranked by cosine similarity.

Design goals
  • Grounded generation: every drafted claim must be traceable to a chunk.
  • Inspectable evidence: callers receive chunk text + relevance score.
  • Isolation: optional doc_id filter keeps retrieval within a single document.
"""

import logging
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from backend.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "legal_documents"


class VectorStore:
    def __init__(self, db_path: Optional[str] = None):
        path = db_path or settings.CHROMA_DB_PATH
        self._client = chromadb.PersistentClient(path=path)
        self._ef = embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_document(
        self,
        doc_id: str,
        chunks: list[str],
        metadata: Optional[dict] = None,
    ) -> None:
        """Embed and store all chunks for a document."""
        if not chunks:
            logger.warning("add_document called with no chunks for doc %s", doc_id)
            return

        ids = [f"{doc_id}__chunk_{i}" for i in range(len(chunks))]
        metas = [
            {"doc_id": doc_id, "chunk_index": i, **(metadata or {})}
            for i in range(len(chunks))
        ]

        # ChromaDB handles batching internally
        self._collection.upsert(ids=ids, documents=chunks, metadatas=metas)
        logger.info("Stored %d chunks for doc %s", len(chunks), doc_id)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Return the top-k most relevant chunks for *query*.

        Each result dict contains:
          text, doc_id, chunk_index, relevance_score (0–1), metadata
        """
        top_k = top_k or settings.TOP_K
        total = self._collection.count()

        if total == 0:
            return []

        where = {"doc_id": doc_id} if doc_id else None
        n_results = min(top_k, total)

        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[dict] = []
        if results["documents"] and results["documents"][0]:
            for text, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append(
                    {
                        "text": text,
                        "doc_id": meta.get("doc_id"),
                        "chunk_index": meta.get("chunk_index"),
                        "relevance_score": round(max(0.0, 1.0 - dist), 4),
                        "metadata": meta,
                    }
                )

        return chunks

    # ── Deletion ──────────────────────────────────────────────────────────────

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks belonging to doc_id. Returns count deleted."""
        existing = self._collection.get(where={"doc_id": doc_id})
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info("Deleted %d chunks for doc %s", len(ids), doc_id)
        return len(ids)

    def document_chunk_count(self, doc_id: str) -> int:
        res = self._collection.get(where={"doc_id": doc_id})
        return len(res.get("ids", []))


# ── Module-level singleton ────────────────────────────────────────────────────

_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
