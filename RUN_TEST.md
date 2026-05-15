# Run, Test, and Real-World Usage Guide

This project is a FastAPI legal-document ingestion, retrieval, and grounded draft
generation app. It uses local OCR/vector storage plus Groq for hosted LLM calls.

## 1. Local Setup

Create a virtual environment and install all runtime + test dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

For LLM draft generation, set at least one Groq key:

```env
GROQ_API_KEY=gsk_key_1
GROQ_API_KEYS=gsk_key_2,gsk_key_3,gsk_key_4
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MODEL_FALLBACKS=llama-3.1-8b-instant
```

For OCR on scanned PDFs/images, install Tesseract on the machine and make sure
`tesseract` is on `PATH`. If it is not, set `TESSERACT_CMD` in `.env`.

## 2. Run All Tests

Basic full test suite:

```bash
python -m pytest tests/ -v --tb=short
```

In locked-down environments where the normal home directory is not writable,
ChromaDB may fail while caching its default ONNX embedding model. Use a writable
temporary home:

```bash
mkdir -p /tmp/psl-test-home
HOME=/tmp/psl-test-home python -m pytest tests/ -v --tb=short
```

The first retrieval-test run may download ChromaDB's default embedding model.
After that, it is cached under the selected `HOME`.

## 3. Extra Checks

Compile all Python files:

```bash
python -m compileall backend evaluate.py
```

Run the evaluation script:

```bash
python evaluate.py
```

Notes:

- Without Groq keys, `evaluate.py` still tests document processing and retrieval,
  then skips LLM draft-generation steps.
- With Groq keys, it also generates a grounded draft and exercises the
  improvement loop.

## 4. Test Result From This Environment

I ran:

```bash
HOME=/tmp/psl-test-home .venv/bin/python -m pytest tests/ -v --tb=short
```

Result:

```text
36 passed in 43.27s
```

Before setting a writable `HOME` and allowing the Chroma model download, the
retrieval tests failed because Chroma tried to write its cache to a read-only
`~/.cache/chroma` path and then could not download the embedding model without
network access.

## 5. Run The App Locally

Start the server:

```bash
./run.sh server
```

Or directly:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://localhost:8000
```

API docs:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/api/health
```

## 6. Run With Docker

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY / GROQ_API_KEYS
./run.sh docker-up
```

Stop Docker:

```bash
./run.sh docker-down
```

## 7. Real-World Inputs

The system expects legal or legal-adjacent documents:

- Native PDFs with selectable text
- Scanned PDFs
- Images such as PNG/JPG/TIFF/WebP
- Plain text or Markdown files

Examples:

- Civil complaint
- Contract or service agreement
- Deposition transcript
- Notice letter
- Title-review packet
- Internal case notes

## 8. Real-World Processing Flow

1. Upload a file in the browser or through `POST /api/documents/upload`.
2. The server saves the file under `data/uploads`.
3. A background task extracts text:
   - PyMuPDF for native PDFs
   - Tesseract OCR for scanned pages/images
   - direct reading for text/Markdown
4. Text is cleaned and chunked.
5. Structured fields are extracted:
   - Groq JSON extraction if keys are configured
   - regex fallback if no Groq key is available
6. Chunks are embedded and stored in ChromaDB under `data/chroma`.
7. Document metadata is stored in SQLite at `data/psl.db`.
8. You generate a draft from the processed document.
9. Retrieval selects the most relevant evidence chunks.
10. Groq generates a grounded draft with `[Evidence N]` citations.
11. You can edit the draft; the system extracts a reusable improvement pattern.
12. Future drafts of the same type receive those learned patterns in the prompt.

## 9. Real-World Outputs

Document upload returns:

```json
{
  "id": "document-uuid",
  "filename": "complaint.pdf",
  "status": "processing"
}
```

Document detail includes:

```json
{
  "id": "document-uuid",
  "filename": "complaint.pdf",
  "file_type": "pdf",
  "raw_text": "cleaned extracted text...",
  "structured_fields": {
    "case_number": "CV-2024-08891",
    "document_type": "Complaint",
    "parties": {
      "plaintiff": "Pinnacle Tech Solutions, Inc.",
      "defendant": "Quantum Dynamics LLC"
    }
  },
  "chunk_count": 8,
  "processing_status": "done"
}
```

Draft generation returns:

```json
{
  "draft_id": "draft-uuid",
  "draft_type": "case_fact_summary",
  "content": "## Parties\nPinnacle is identified as the plaintiff [Evidence 1]...",
  "evidence_chunks": [
    {
      "chunk_index": 0,
      "relevance_score": 0.82,
      "text_preview": "SUPERIOR COURT OF THE STATE OF NEW YORK..."
    }
  ],
  "patterns_applied": 0,
  "created_at": "2026-05-15 12:00:00"
}
```

The user-facing output is a legal-style draft with cited evidence, plus a side
panel showing the evidence chunks used.

## 10. Useful API Commands

Upload a document:

```bash
curl -F "file=@sample_docs/case_001_complaint.txt" \
  http://localhost:8000/api/documents/upload
```

List documents:

```bash
curl http://localhost:8000/api/documents
```

Check document status:

```bash
curl http://localhost:8000/api/documents/<document_id>/status
```

Generate a draft:

```bash
curl -X POST http://localhost:8000/api/drafts/generate \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "<document_id>",
    "draft_type": "case_fact_summary"
  }'
```

Inspect retrieval:

```bash
curl -X POST http://localhost:8000/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "<document_id>",
    "query": "key parties claims damages deadlines",
    "top_k": 5
  }'
```

Submit an edited draft:

```bash
curl -X POST http://localhost:8000/api/drafts/<draft_id>/edit \
  -H "Content-Type: application/json" \
  -d '{
    "edited_content": "Corrected draft text goes here..."
  }'
```
