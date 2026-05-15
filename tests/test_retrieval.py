"""
Unit tests for retrieval.py.

Uses an isolated ChromaDB instance (via the vector_store fixture) so the
production index is never touched.
"""

import pytest


# ── add_document + retrieve ───────────────────────────────────────────────────

class TestVectorStore:
    def test_add_and_retrieve(self, vector_store):
        chunks = [
            "The defendant breached the contract by failing to deliver the software.",
            "Plaintiff paid $1,900,000 in milestone payments prior to the breach.",
            "The filing date was March 15, 2024 in the County of New York.",
        ]
        vector_store.add_document("doc-001", chunks, metadata={"filename": "test.txt"})

        results = vector_store.retrieve("breach of contract payment", top_k=3)
        assert len(results) >= 1
        assert all("text" in r for r in results)
        assert all("relevance_score" in r for r in results)
        assert all(0.0 <= r["relevance_score"] <= 1.0 for r in results)

    def test_doc_id_filter(self, vector_store):
        vector_store.add_document("doc-A", ["Alpha document about contracts."], {})
        vector_store.add_document("doc-B", ["Beta document about torts and negligence."], {})

        results = vector_store.retrieve("contracts", doc_id="doc-A", top_k=5)
        assert all(r["doc_id"] == "doc-A" for r in results)

    def test_empty_store_returns_empty(self, vector_store):
        results = vector_store.retrieve("anything")
        assert results == []

    def test_delete_document(self, vector_store):
        vector_store.add_document("doc-del", ["To be deleted."], {})
        assert vector_store.document_chunk_count("doc-del") == 1

        vector_store.delete_document("doc-del")
        assert vector_store.document_chunk_count("doc-del") == 0

    def test_relevance_scores_ordered(self, vector_store):
        chunks = [
            "Contract signed on January 1 2023 for software delivery.",
            "The weather in New York is cold in January.",
            "Breach of contract damages totalling two million dollars.",
        ]
        vector_store.add_document("doc-score", chunks, {})
        results = vector_store.retrieve("breach contract damages", top_k=3)
        scores = [r["relevance_score"] for r in results]
        # Scores should be descending
        assert scores == sorted(scores, reverse=True)

    def test_upsert_does_not_duplicate(self, vector_store):
        chunks = ["First version of the document text here."]
        vector_store.add_document("doc-upsert", chunks, {})
        vector_store.add_document("doc-upsert", chunks, {})
        assert vector_store.document_chunk_count("doc-upsert") == 1
