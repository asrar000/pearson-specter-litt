"""
FastAPI application — Pearson Specter Litt AI Legal System.

Endpoints
─────────
  POST   /api/documents/upload          Upload + process a document (background)
  GET    /api/documents                 List all documents
  GET    /api/documents/{id}            Full document detail (text + fields)
  DELETE /api/documents/{id}            Delete document + vector data
  GET    /api/documents/{id}/status     Lightweight status polling

  POST   /api/drafts/generate           Generate a grounded draft
  GET    /api/drafts/{id}               Retrieve a draft
  POST   /api/drafts/{id}/edit          Submit an operator edit
  GET    /api/documents/{id}/drafts     Draft history for a document

  POST   /api/retrieve                  Inspect retrieved evidence for a query

  GET    /api/patterns                  List learned patterns (?draft_type=)
  DELETE /api/patterns/{id}             Remove a pattern

  GET    /api/health                    Health check
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import get_db, init_db
from backend.document_processor import process_document
from backend.draft_generator import DRAFT_TYPES, generate_draft
from backend.improvement_loop import get_relevant_patterns, save_edit_and_extract_pattern
from backend.models import Document, Draft, LearnedPattern, OperatorEdit
from backend.retrieval import get_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pearson Specter Litt AI",
    description="Grounded legal document drafting system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    init_db()
    logger.info("Database initialised at %s", settings.SQLITE_DB_PATH)


# ── Request / Response schemas ────────────────────────────────────────────────


class GenerateDraftRequest(BaseModel):
    document_id: str
    draft_type: str = "case_fact_summary"
    query: Optional[str] = None


class SubmitEditRequest(BaseModel):
    edited_content: str


class RetrieveRequest(BaseModel):
    query: str
    document_id: Optional[str] = None
    top_k: int = 6


# ── Background processing ─────────────────────────────────────────────────────


def _process_document_bg(doc_id: str, file_path: str) -> None:
    """
    Background task — runs in a thread with its own DB session.

    Processes the document, stores results in SQLite, and adds chunks to
    the vector store.
    """
    from backend.database import SessionLocal

    db = SessionLocal()
    try:
        doc: Optional[Document] = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return

        doc.processing_status = "processing"
        db.commit()

        result = process_document(file_path)

        doc.raw_text = result["raw_text"]
        doc.file_type = result["file_type"]
        doc.structured_fields = result["structured_fields"]
        doc.chunk_count = result["chunk_count"]
        doc.processing_status = "done"
        db.commit()

        vs = get_vector_store()
        vs.add_document(
            doc_id=doc_id,
            chunks=result["chunks"],
            metadata={
                "filename": doc.filename,
                "doc_type": (result["structured_fields"] or {}).get("document_type", "unknown"),
            },
        )
        logger.info("Document %s ready (%d chunks)", doc_id, result["chunk_count"])

    except Exception as exc:
        logger.exception("Processing failed for doc %s", doc_id)
        try:
            db.rollback()
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.processing_status = "failed"
                doc.error_message = str(exc)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


# ── Document endpoints ────────────────────────────────────────────────────────


@app.post("/api/documents/upload", tags=["Documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a legal document for processing (PDF, image, or text)."""
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}_{file.filename}"
    file_path = upload_dir / safe_name

    with open(file_path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    doc = Document(
        id=doc_id,
        filename=file.filename,
        file_path=str(file_path),
        processing_status="pending",
    )
    db.add(doc)
    db.commit()

    background_tasks.add_task(_process_document_bg, doc_id, str(file_path))

    return {"id": doc_id, "filename": file.filename, "status": "processing"}


@app.get("/api/documents", tags=["Documents"])
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "chunk_count": d.chunk_count,
            "processing_status": d.processing_status,
            "structured_fields": d.structured_fields,
            "created_at": str(d.created_at),
        }
        for d in docs
    ]


@app.get("/api/documents/{doc_id}", tags=["Documents"])
def get_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": doc.id,
        "filename": doc.filename,
        "file_type": doc.file_type,
        "raw_text": doc.raw_text,
        "structured_fields": doc.structured_fields,
        "chunk_count": doc.chunk_count,
        "processing_status": doc.processing_status,
        "error_message": doc.error_message,
        "created_at": str(doc.created_at),
    }


@app.get("/api/documents/{doc_id}/status", tags=["Documents"])
def get_document_status(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": doc.id,
        "processing_status": doc.processing_status,
        "chunk_count": doc.chunk_count,
        "error_message": doc.error_message,
    }


@app.delete("/api/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    vs = get_vector_store()
    deleted_chunks = vs.delete_document(doc_id)

    db.delete(doc)
    db.commit()

    return {"message": f"Document deleted ({deleted_chunks} chunks removed from index)"}


# ── Draft endpoints ───────────────────────────────────────────────────────────


@app.post("/api/drafts/generate", tags=["Drafts"])
def generate_draft_endpoint(req: GenerateDraftRequest, db: Session = Depends(get_db)):
    """Generate a grounded draft for a processed document."""
    doc = db.query(Document).filter(Document.id == req.document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.processing_status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Document not ready (status: {doc.processing_status}). "
            "Poll /api/documents/{id}/status until status='done'.",
        )
    if req.draft_type not in DRAFT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown draft type '{req.draft_type}'. "
            f"Valid options: {list(DRAFT_TYPES.keys())}",
        )
    if not settings.GROQ_API_KEYS:
        raise HTTPException(
            status_code=503,
            detail="Groq API key not configured. Set GROQ_API_KEY or GROQ_API_KEYS.",
        )

    vs = get_vector_store()

    # Build a retrieval query from the request or the draft type label
    query = req.query or (
        f"Key facts, parties, dates, claims, and legal issues relevant to a "
        f"{DRAFT_TYPES[req.draft_type]} for {doc.filename}"
    )

    # Try document-scoped retrieval first; fall back to global if no hits
    chunks = vs.retrieve(query=query, doc_id=req.document_id)
    if not chunks:
        chunks = vs.retrieve(query=query)

    doc_metadata = {
        "structured_fields": doc.structured_fields,
        "filename": doc.filename,
    }

    result = generate_draft(
        document_id=req.document_id,
        doc_metadata=doc_metadata,
        retrieved_chunks=chunks,
        draft_type=req.draft_type,
        db=db,
    )

    draft = Draft(
        document_id=req.document_id,
        draft_type=req.draft_type,
        content=result["content"],
        evidence_chunks=result["evidence_chunks"],
        prompt_used=result["prompt_used"],
        patterns_applied=result["patterns_applied"],
    )
    db.add(draft)
    db.commit()
    db.refresh(draft)

    return {
        "draft_id": draft.id,
        "draft_type": draft.draft_type,
        "content": draft.content,
        "evidence_chunks": draft.evidence_chunks,
        "patterns_applied": draft.patterns_applied,
        "created_at": str(draft.created_at),
    }


@app.get("/api/drafts/{draft_id}", tags=["Drafts"])
def get_draft(draft_id: str, db: Session = Depends(get_db)):
    draft = db.query(Draft).filter(Draft.id == draft_id).first()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    return {
        "draft_id": draft.id,
        "document_id": draft.document_id,
        "draft_type": draft.draft_type,
        "content": draft.content,
        "evidence_chunks": draft.evidence_chunks,
        "patterns_applied": draft.patterns_applied,
        "created_at": str(draft.created_at),
    }


@app.post("/api/drafts/{draft_id}/edit", tags=["Drafts"])
def submit_edit(draft_id: str, req: SubmitEditRequest, db: Session = Depends(get_db)):
    """Submit an operator correction. The system extracts a reusable pattern."""
    draft = db.query(Draft).filter(Draft.id == draft_id).first()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")

    edit, pattern = save_edit_and_extract_pattern(
        db=db,
        draft_id=draft_id,
        original_content=draft.content,
        edited_content=req.edited_content,
        draft_type=draft.draft_type,
    )

    return {
        "edit_id": edit.id,
        "pattern_learned": edit.extracted_pattern,
        "pattern_id": pattern.id if pattern else None,
        "message": "Edit saved. Pattern extracted and will improve future drafts.",
    }


@app.get("/api/documents/{doc_id}/drafts", tags=["Drafts"])
def list_drafts_for_document(doc_id: str, db: Session = Depends(get_db)):
    drafts = (
        db.query(Draft)
        .filter(Draft.document_id == doc_id)
        .order_by(Draft.created_at.desc())
        .all()
    )
    return [
        {
            "draft_id": d.id,
            "draft_type": d.draft_type,
            "patterns_applied": d.patterns_applied,
            "edit_count": len(d.edits),
            "created_at": str(d.created_at),
        }
        for d in drafts
    ]


# ── Retrieval inspection ──────────────────────────────────────────────────────


@app.post("/api/retrieve", tags=["Retrieval"])
def inspect_retrieval(req: RetrieveRequest, db: Session = Depends(get_db)):
    """
    Retrieve evidence chunks for a query without generating a draft.
    Useful for understanding what evidence would support a claim.
    """
    vs = get_vector_store()
    chunks = vs.retrieve(
        query=req.query,
        top_k=req.top_k,
        doc_id=req.document_id,
    )
    return {
        "query": req.query,
        "doc_id": req.document_id,
        "chunks_returned": len(chunks),
        "chunks": chunks,
    }


# ── Pattern endpoints ─────────────────────────────────────────────────────────


@app.get("/api/patterns", tags=["Patterns"])
def list_patterns(
    draft_type: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(LearnedPattern)
    if draft_type:
        q = q.filter(LearnedPattern.draft_type == draft_type)
    patterns = q.order_by(LearnedPattern.usage_count.desc()).all()
    return [
        {
            "id": p.id,
            "draft_type": p.draft_type,
            "pattern_type": p.pattern_type,
            "pattern_description": p.pattern_description,
            "example_before": p.example_before,
            "example_after": p.example_after,
            "relevance_tags": p.relevance_tags,
            "usage_count": p.usage_count,
            "created_at": str(p.created_at),
        }
        for p in patterns
    ]


@app.delete("/api/patterns/{pattern_id}", tags=["Patterns"])
def delete_pattern(pattern_id: str, db: Session = Depends(get_db)):
    p = db.query(LearnedPattern).filter(LearnedPattern.id == pattern_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Pattern not found")
    db.delete(p)
    db.commit()
    return {"message": "Pattern removed"}


# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/api/health", tags=["System"])
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/draft-types", tags=["System"])
def draft_types():
    return DRAFT_TYPES


# ── Static frontend ───────────────────────────────────────────────────────────
# Mount after all API routes so they take priority

@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")


try:
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
except RuntimeError:
    pass  # frontend dir might not exist in test environments
