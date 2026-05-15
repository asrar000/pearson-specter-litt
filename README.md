# ⚖ Pearson Specter Litt — AI Legal Document System

A grounded legal document processing and draft generation system.
Built for messy real-world inputs: scanned PDFs, OCR noise, handwritten
notes, and inconsistently formatted files.

---

## What It Does

| Stage | What happens |
|---|---|
| **Ingest** | Upload PDF, image, or text. PyMuPDF extracts native text; pytesseract handles scanned/image pages. |
| **Structure** | Claude extracts case number, parties, dates, claims, and value into structured JSON (regex fallback if no API key). |
| **Index** | Cleaned text is chunked with sentence-aware overlap and embedded into ChromaDB. |
| **Retrieve** | A query (auto-generated from draft type) retrieves the top-k most relevant chunks by cosine similarity. |
| **Draft** | Claude generates a grounded draft. Every claim must cite `[Evidence N]`. Unsupported claims are flagged `INSUFFICIENT EVIDENCE`. |
| **Edit** | An operator corrects the draft. The system sends the (original, edited) pair to Claude to distil a reusable improvement pattern. |
| **Improve** | Learned patterns are injected into future prompts as few-shot examples, measurably improving citation discipline and completeness. |

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and on `PATH`
- An Anthropic API key

### Install

```bash
cd pearson-specter-litt
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY at minimum
```

### Run

```bash
./run.sh server
# or: uvicorn backend.main:app --reload
```

Open **http://localhost:8000** in your browser.

### Run Tests

```bash
./run.sh test
# or: python -m pytest tests/ -v
```

### Run Evaluation

```bash
./run.sh evaluate
# or: python evaluate.py
```

---

## Docker

```bash
cp .env.example .env        # set ANTHROPIC_API_KEY
./run.sh docker-up
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/documents/upload` | Upload a document |
| `GET` | `/api/documents` | List all documents |
| `GET` | `/api/documents/{id}` | Document detail + text |
| `GET` | `/api/documents/{id}/status` | Lightweight status poll |
| `DELETE` | `/api/documents/{id}` | Delete doc + vectors |
| `POST` | `/api/drafts/generate` | Generate grounded draft |
| `GET` | `/api/drafts/{id}` | Get a draft |
| `POST` | `/api/drafts/{id}/edit` | Submit operator correction |
| `GET` | `/api/documents/{id}/drafts` | Draft history for a doc |
| `POST` | `/api/retrieve` | Inspect retrieved evidence |
| `GET` | `/api/patterns` | List learned patterns |
| `DELETE` | `/api/patterns/{id}` | Remove a pattern |

Interactive docs at **http://localhost:8000/docs**.

---

## Sample Documents

Three synthetic legal documents are provided in `sample_docs/`:

| File | Content |
|---|---|
| `case_001_complaint.txt` | Civil complaint — breach of contract + fraud |
| `case_001_contract.txt` | The underlying Software Services Agreement |
| `case_002_deposition.txt` | Deposition transcript of the defendant |

---

## Draft Types

| Key | Label |
|---|---|
| `case_fact_summary` | Case Fact Summary |
| `internal_memo` | First-Pass Internal Memo |
| `notice_summary` | Notice-Related Summary |
| `title_review` | Title Review Summary |
| `document_checklist` | Document Checklist |

---

## Project Structure

```
pearson-specter-litt/
├── backend/
│   ├── config.py            Settings + env vars
│   ├── database.py          SQLAlchemy engine + session factory
│   ├── models.py            Document, Draft, OperatorEdit, LearnedPattern
│   ├── document_processor.py  OCR, extraction, cleaning, chunking
│   ├── retrieval.py         ChromaDB vector store wrapper
│   ├── draft_generator.py   Grounded draft generation with citation enforcement
│   ├── improvement_loop.py  Edit capture + pattern distillation
│   └── main.py              FastAPI application
├── frontend/
│   └── index.html           Single-page UI (vanilla JS)
├── sample_docs/             Synthetic legal documents
├── tests/                   pytest test suite
├── evaluate.py              End-to-end evaluation script
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Evaluation Approach

Run `python evaluate.py` to get a full report covering:

1. **Processing stats**: file type, char count, chunk count, field extraction
2. **Retrieval quality**: relevance scores for test queries
3. **Grounding score**: % of substantive paragraphs with `[Evidence N]` citations
4. **Improvement delta**: grounding score before vs. after pattern injection

The grounding score is the primary quality metric: a well-grounded draft cites
evidence after every factual claim and marks gaps explicitly.
