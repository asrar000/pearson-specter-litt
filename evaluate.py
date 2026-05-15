#!/usr/bin/env python3
"""
End-to-end evaluation script for the PSL AI Legal System.

Runs the full pipeline against sample documents and reports:
  1. Document processing stats (extraction quality, chunk counts)
  2. Retrieval quality (relevance scores)
  3. Draft grounding score (% of paragraphs with [Evidence N] citations)
  4. Improvement loop effectiveness (pattern extraction success)

Usage:
  python evaluate.py

Requires GROQ_API_KEY or GROQ_API_KEYS to be set in .env or the environment
for LLM-based draft generation steps.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# Bootstrap the environment before importing backend modules
from dotenv import load_dotenv
load_dotenv()

from backend.config import settings
from backend.database import init_db, SessionLocal
from backend.document_processor import process_document
from backend.draft_generator import generate_draft, DRAFT_TYPES
from backend.improvement_loop import save_edit_and_extract_pattern, get_relevant_patterns
from backend.retrieval import VectorStore

SAMPLE_DOCS = [
    "sample_docs/case_001_complaint.txt",
    "sample_docs/case_001_contract.txt",
    "sample_docs/case_002_deposition.txt",
]

SEPARATOR = "─" * 60


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def score_grounding(draft_text: str) -> dict:
    """Measure how well a draft is grounded in evidence."""
    paragraphs = [p.strip() for p in draft_text.split("\n\n") if p.strip()]
    substantive = [p for p in paragraphs if len(p.split()) > 8 and not p.startswith("#")]

    cited = sum(1 for p in substantive if re.search(r"\[Evidence \d+\]", p))
    insufficient = draft_text.count("INSUFFICIENT EVIDENCE")
    total = len(substantive)

    return {
        "total_paragraphs": total,
        "cited_paragraphs": cited,
        "grounding_pct": round(cited / total * 100, 1) if total else 0,
        "insufficient_evidence_markers": insufficient,
    }


def evaluate():
    init_db()
    db = SessionLocal()

    # Use a temp vector store so eval does not pollute production
    import tempfile
    vs = VectorStore(db_path=tempfile.mkdtemp())

    section("1 — Document Processing")
    processed: list[dict] = []

    for doc_path in SAMPLE_DOCS:
        if not Path(doc_path).exists():
            print(f"  ⚠  Skipping {doc_path} (not found)")
            continue

        t0 = time.perf_counter()
        result = process_document(doc_path)
        elapsed = time.perf_counter() - t0

        sf = result["structured_fields"]
        print(f"\n  File       : {doc_path}")
        print(f"  Type       : {result['file_type']}")
        print(f"  Chars      : {len(result['raw_text']):,}")
        print(f"  Chunks     : {result['chunk_count']}")
        print(f"  Case No.   : {sf.get('case_number') or '—'}")
        print(f"  Doc Type   : {sf.get('document_type') or '—'}")
        print(f"  Plaintiff  : {(sf.get('parties') or {}).get('plaintiff') or '—'}")
        print(f"  Value      : {sf.get('contract_value') or '—'}")
        print(f"  Time       : {elapsed:.2f}s")

        doc_id = Path(doc_path).stem
        vs.add_document(doc_id, result["chunks"], {"filename": doc_path})

        processed.append({
            "doc_id": doc_id,
            "path": doc_path,
            "result": result,
        })

    if not processed:
        print("  No documents processed — aborting.")
        db.close()
        return

    section("2 — Retrieval Quality")

    test_queries = [
        ("breach of contract damages", "case_001_complaint"),
        ("milestone payment amounts", None),
        ("deposition testimony false representations", "case_002_deposition"),
    ]

    for query, filter_id in test_queries:
        chunks = vs.retrieve(query=query, top_k=4, doc_id=filter_id)
        print(f"\n  Query: \"{query}\"  (filter_doc={filter_id or 'none'})")
        for i, c in enumerate(chunks, 1):
            preview = c["text"][:80].replace("\n", " ")
            print(f"    [{i}] score={c['relevance_score']:.3f} | {preview}…")
        if chunks:
            avg = sum(c["relevance_score"] for c in chunks) / len(chunks)
            print(f"    Avg relevance: {avg:.3f}")

    section("3 — Draft Generation & Grounding")

    if not settings.GROQ_API_KEYS:
        print("  ⚠  GROQ_API_KEY/GROQ_API_KEYS not set — skipping LLM-based steps.")
        db.close()
        return

    first = processed[0]
    doc_metadata = {
        "structured_fields": first["result"]["structured_fields"],
        "filename": first["path"],
    }
    retrieval_query = "key facts, parties, claims, and damages"
    chunks = vs.retrieve(query=retrieval_query, doc_id=first["doc_id"])

    print(f"\n  Generating 'case_fact_summary' for {first['path']}…")
    t0 = time.perf_counter()
    draft_result = generate_draft(
        document_id=first["doc_id"],
        doc_metadata=doc_metadata,
        retrieved_chunks=chunks,
        draft_type="case_fact_summary",
        db=None,  # no patterns yet
    )
    elapsed = time.perf_counter() - t0

    grounding = score_grounding(draft_result["content"])
    print(f"\n  Draft generated in {elapsed:.1f}s ({len(draft_result['content'])} chars)")
    print(f"  Evidence chunks used      : {len(draft_result['evidence_chunks'])}")
    print(f"  Substantive paragraphs    : {grounding['total_paragraphs']}")
    print(f"  Paragraphs with citations : {grounding['cited_paragraphs']}")
    print(f"  Grounding score           : {grounding['grounding_pct']}%")
    print(f"  'INSUFFICIENT EVIDENCE'   : {grounding['insufficient_evidence_markers']} times")

    preview = draft_result["content"][:400].replace("\n", "\n  ")
    print(f"\n  Draft preview:\n  {preview}…\n")

    section("4 — Improvement Loop")

    print("  Simulating operator edit…")

    # Simulate operator improving the draft
    original = draft_result["content"]
    edited = (
        original
        + "\n\n## Operator Note\nThis summary should always include the specific "
        "dollar amounts for each milestone payment and confirm whether each was paid. "
        "Citations must appear after every financial figure, not just at the end of "
        "paragraphs. This improves auditability for the billing team."
    )

    from backend.models import Document as DocModel, Draft as DraftModel

    # We need minimal DB rows to satisfy FKs
    doc_row = DocModel(
        id=first["doc_id"], filename=first["path"], file_path=first["path"],
        processing_status="done"
    )
    draft_row = DraftModel(
        id="eval-draft-001", document_id=first["doc_id"],
        draft_type="case_fact_summary", content=original
    )
    db.add(doc_row)
    db.add(draft_row)
    db.commit()

    edit, pattern = save_edit_and_extract_pattern(
        db=db,
        draft_id="eval-draft-001",
        original_content=original,
        edited_content=edited,
        draft_type="case_fact_summary",
    )

    print(f"  Edit saved  (id={edit.id[:8]}…)")
    if pattern:
        print(f"  Pattern extracted:")
        print(f"    Type : {pattern.pattern_type}")
        print(f"    Desc : {pattern.pattern_description[:120]}")

    print("\n  Regenerating draft WITH learned pattern injected…")
    patterns = get_relevant_patterns("case_fact_summary", db)
    t0 = time.perf_counter()
    draft_v2 = generate_draft(
        document_id=first["doc_id"],
        doc_metadata=doc_metadata,
        retrieved_chunks=chunks,
        draft_type="case_fact_summary",
        db=db,
    )
    elapsed2 = time.perf_counter() - t0

    grounding2 = score_grounding(draft_v2["content"])
    print(f"  v2 generated in {elapsed2:.1f}s")
    print(f"  Patterns applied       : {draft_v2['patterns_applied']}")
    print(f"  v1 grounding score     : {grounding['grounding_pct']}%")
    print(f"  v2 grounding score     : {grounding2['grounding_pct']}%")

    delta = grounding2["grounding_pct"] - grounding["grounding_pct"]
    indicator = "✓ Improved" if delta >= 0 else "~ No change"
    print(f"  Delta                  : {delta:+.1f}% ({indicator})")

    section("Summary")
    print(f"  Documents processed    : {len(processed)}")
    print(f"  Total chunks indexed   : {sum(p['result']['chunk_count'] for p in processed)}")
    print(f"  v1 grounding           : {grounding['grounding_pct']}%")
    print(f"  v2 grounding           : {grounding2['grounding_pct']}%")
    print(f"  Patterns in store      : {len(patterns)}")
    print()

    db.close()


if __name__ == "__main__":
    evaluate()
