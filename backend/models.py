import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Document(Base):
    """A processed legal document stored in the system."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String)  # pdf | image | text
    raw_text = Column(Text)
    structured_fields = Column(JSON)  # extracted metadata
    chunk_count = Column(Integer, default=0)
    processing_status = Column(String, default="pending")  # pending|processing|done|failed
    error_message = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

    drafts = relationship("Draft", back_populates="document", cascade="all, delete-orphan")


class Draft(Base):
    """An AI-generated draft grounded in a document."""

    __tablename__ = "drafts"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    draft_type = Column(String, default="case_fact_summary")
    content = Column(Text)
    evidence_chunks = Column(JSON)  # serialised list of {chunk_index, score, text_preview}
    prompt_used = Column(Text)
    patterns_applied = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())

    document = relationship("Document", back_populates="drafts")
    edits = relationship("OperatorEdit", back_populates="draft", cascade="all, delete-orphan")


class OperatorEdit(Base):
    """A human correction made to a draft."""

    __tablename__ = "operator_edits"

    id = Column(String, primary_key=True, default=_uuid)
    draft_id = Column(String, ForeignKey("drafts.id"), nullable=False)
    original_content = Column(Text)
    edited_content = Column(Text)
    extracted_pattern = Column(JSON)  # pattern distilled by the improvement loop
    created_at = Column(DateTime, server_default=func.now())

    draft = relationship("Draft", back_populates="edits")


class LearnedPattern(Base):
    """A reusable improvement pattern extracted from operator edits."""

    __tablename__ = "learned_patterns"

    id = Column(String, primary_key=True, default=_uuid)
    draft_type = Column(String, index=True)
    pattern_description = Column(Text)
    pattern_type = Column(String)  # tone|structure|completeness|accuracy|citation|formatting
    example_before = Column(Text)
    example_after = Column(Text)
    relevance_tags = Column(JSON)
    usage_count = Column(Integer, default=1)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
