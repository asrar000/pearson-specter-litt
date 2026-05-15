# Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser / API)                │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼────────────────────────────────┐
│                      FastAPI Application                     │
│                         (backend/main.py)                    │
└───┬───────────────┬──────────────────────┬──────────────────┘
    │               │                      │
    ▼               ▼                      ▼
┌───────────┐ ┌───────────┐        ┌──────────────┐
│ Document  │ │  Retrieval│        │  Improvement │
│ Processor │ │  Layer    │        │  Loop        │
│           │ │           │        │              │
│ PyMuPDF   │ │ ChromaDB  │        │ Edit capture │
│ Tesseract │ │ (vectors) │        │ Pattern      │
│ Claude    │ │           │        │ extraction   │
│ (fields)  │ │           │        │ (Claude)     │
└─────┬─────┘ └─────┬─────┘        └──────┬───────┘
      │             │                     │
      ▼             ▼                     ▼
┌─────────────────────────────────────────────────┐
│                    SQLite Database               │
│  Document | Draft | OperatorEdit | LearnedPattern│
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────┐
│  Draft Generator│
│  (Claude API)   │
│                 │
│  Evidence →     │
│  Grounded Draft │
│  with [Ev N]    │
│  citations      │
└─────────────────┘
```

## Component Responsibilities

### Document Processor (`document_processor.py`)

Handles messy inputs in three stages:

1. **Extraction**: PyMuPDF for native PDF text. Per-page OCR fallback (Tesseract
   at 300dpi) when a page yields fewer than 40 tokens. Standalone image OCR.
   Plain text / Markdown read directly.

2. **Cleaning**: Collapses excessive whitespace, strips control characters,
   fixes common OCR substitutions (l→1, |→I in legal contexts).

3. **Chunking**: Sentence-boundary-aware splitting with configurable size and
   overlap. Overlap preserves cross-chunk context for retrieval.

4. **Field extraction**: Claude extracts case number, parties, dates, jurisdiction,
   claims, and contract value into JSON. Regex fallback when the API is unavailable.

### Retrieval Layer (`retrieval.py`)

A thin wrapper around ChromaDB with a cosine-similarity index.

- Documents are stored as overlapping chunks (not full documents).
- Retrieval can be scoped to a single document (`doc_id` filter) or global.
- Returns chunks with relevance scores (0–1) for transparency.
- The retrieval query is derived from the draft type + optional user query.

### Draft Generator (`draft_generator.py`)

Generates drafts using a two-part prompt:

**System prompt** enforces grounding rules:
- Every factual claim must end with `[Evidence N]`.
- Unsupported claims must be marked `INSUFFICIENT EVIDENCE`.
- No external knowledge without explicit labelling.

**User turn** contains:
- Structured metadata (case number, parties, etc.)
- Numbered evidence block from the retrieval layer

Learned patterns from prior operator edits are appended to the system prompt
as few-shot examples, guiding future output toward operator-preferred style.

### Improvement Loop (`improvement_loop.py`)

The improvement loop is **semantic**, not structural:

1. Operator submits an edited draft.
2. The (original, edited) pair is sent to Claude with the prompt:
   *"What is the key improvement pattern here?"*
3. Claude returns a JSON pattern: `{description, type, before, after, tags}`.
4. Patterns are stored in `LearnedPattern` and deduplicated by description.
5. On the next draft generation for the same draft type, the top-N patterns
   (ordered by usage count) are injected into the system prompt.

This is a real improvement loop: the model changes its behaviour based on
accumulated operator feedback, not just a stored diff.

## Data Model

```
Document
  id, filename, file_path, file_type
  raw_text (cleaned), structured_fields (JSON)
  chunk_count, processing_status

Draft
  id, document_id → Document
  draft_type, content, evidence_chunks (JSON)
  prompt_used, patterns_applied

OperatorEdit
  id, draft_id → Draft
  original_content, edited_content
  extracted_pattern (JSON)

LearnedPattern
  id, draft_type
  pattern_description, pattern_type
  example_before, example_after
  relevance_tags (JSON), usage_count
```

## Key Design Decisions & Tradeoffs

| Decision | Rationale | Tradeoff |
|---|---|---|
| SQLite for metadata | Zero-ops, no external dependencies | Single-writer; fine for this scale |
| ChromaDB for vectors | Persistent, no external service | Less scalable than Pinecone/Weaviate |
| Sentence-aware chunking | Better coherence than fixed-char splits | Slightly longer chunks possible |
| Background document processing | Non-blocking upload response | Poll `/status` needed |
| `[Evidence N]` citations enforced in prompt | Enables grounding inspection | LLM compliance not 100% guaranteed |
| Pattern dedup by description | Avoids redundant examples | Semantic duplicates with different text may still be stored |
| Regex fallback for field extraction | Works without API key | Lower accuracy on edge cases |

## Scalability Notes

The system is designed for single-server deployment with moderate document loads.
To scale:

- Replace SQLite with PostgreSQL (swap the SQLAlchemy connection string).
- Replace ChromaDB with a managed vector store (Pinecone, Weaviate).
- Move background processing to Celery + Redis for parallelism.
- Add caching for repeated retrieval queries (Redis).
- The core pipeline logic (processor, retrieval, generator, loop) is
  stateless and can be extracted into microservices.
