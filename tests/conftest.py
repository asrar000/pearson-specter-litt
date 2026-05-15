"""
Shared fixtures for the PSL AI test suite.

Key design decisions:
  • All database fixtures use in-memory or tmp_path SQLite — no real DB touched.
  • ChromaDB fixtures use tmp_path to avoid polluting the production store.
  • The Groq API is not called in unit tests; LLM-dependent paths use fallbacks.
"""

import os
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Point config at test paths BEFORE importing any backend modules
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("GROQ_API_KEYS", "")
os.environ.setdefault("CHROMA_DB_PATH", tempfile.mkdtemp())
os.environ.setdefault("SQLITE_DB_PATH", ":memory:")
os.environ.setdefault("UPLOAD_DIR", tempfile.mkdtemp())


# ── Database fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def db_session(tmp_path):
    """In-memory SQLAlchemy session with all tables created."""
    from backend.database import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


# ── Sample text fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_legal_text():
    return (
        "SUPERIOR COURT OF THE STATE OF NEW YORK\n"
        "Case No.: CV-2024-08891\n"
        "Filed: March 15, 2024\n\n"
        "PINNACLE TECH SOLUTIONS, INC., Plaintiff,\n"
        "v.\n"
        "QUANTUM DYNAMICS LLC, Defendant.\n\n"
        "Plaintiff alleges breach of the Software Services Agreement dated "
        "September 1, 2023. Total contract value was $2,850,000. Plaintiff paid "
        "$1,900,000 in milestone payments. Defendant failed to deliver the System "
        "by the January 31, 2024 deadline. Plaintiff demands compensatory damages "
        "of no less than $5,550,000 and punitive damages.\n"
    )


@pytest.fixture
def sample_txt_file(tmp_path, sample_legal_text):
    f = tmp_path / "test_case.txt"
    f.write_text(sample_legal_text)
    return str(f)


# ── Vector store fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def vector_store(tmp_path):
    """Isolated VectorStore backed by a temporary ChromaDB directory."""
    from backend.retrieval import VectorStore
    return VectorStore(db_path=str(tmp_path / "chroma"))
