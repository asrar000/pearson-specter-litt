# ⚖ Pearson Specter Litt — AI Legal Document System

A grounded legal document processing and draft generation system.
Built for messy real-world inputs: scanned PDFs, OCR noise, handwritten
notes, and inconsistently formatted files.

---

## What It Does

| Stage | What happens |
|---|---|
| **Ingest** | Upload PDF, image, or text. PyMuPDF extracts native text; pytesseract handles scanned/image pages. |
| **Structure** | Groq extracts case number, parties, dates, claims, and value into structured JSON (regex fallback if no API key). |
| **Index** | Cleaned text is chunked with sentence-aware overlap and embedded into ChromaDB. |
| **Retrieve** | A query (auto-generated from draft type) retrieves the top-k most relevant chunks by cosine similarity. |
| **Draft** | Groq generates a grounded draft. Every claim must cite `[Evidence N]`. Unsupported claims are flagged `INSUFFICIENT EVIDENCE`. |
| **Edit** | An operator corrects the draft. The system sends the (original, edited) pair to Groq to distil a reusable improvement pattern. |
| **Improve** | Learned patterns are injected into future prompts as few-shot examples, measurably improving citation discipline and completeness. |

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and on `PATH`
- A [Groq API key](https://console.groq.com/keys) for draft generation

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
# Edit .env — set GROQ_API_KEY or GROQ_API_KEYS
```

### Groq Key Pool

You can use one Groq key:

```bash
GROQ_API_KEY=gsk_primary_key
```

Or rotate across several keys:

```bash
GROQ_API_KEY=gsk_key_1
GROQ_API_KEYS=gsk_key_2,gsk_key_3,gsk_key_4
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MODEL_FALLBACKS=llama-3.1-8b-instant
GROQ_MAX_TOKENS=420
```

The app automatically rotates keys on auth, rate-limit, transient network, and
server errors. If the main model is unavailable, it tries fallback models in
order. Increase `GROQ_MAX_TOKENS` if generated drafts feel too short.

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
cp .env.example .env        # set GROQ_API_KEY or GROQ_API_KEYS
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

## Sample Input / Output

`sample_input/` contains text files that can be treated as source documents for processing and draft generation, rather than pre-defined prompt labels. Each file in `sample_input/` may be ingested, structured, chunked, and indexed exactly like files uploaded through the API.

Generated results can be saved under `sample_output/` as:
- `*_processed.json` — extracted text, metadata, and chunk details
- `*_draft.md` — generated draft content
- `*_evidence.txt` — numbered retrieved evidence blocks

This is useful for batch experiments, reproducible examples, or environments where you want to skip the web upload step.

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
