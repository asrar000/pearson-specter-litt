"""
Document processing pipeline.

Handles:
  • Native PDF text extraction (PyMuPDF)
  • Per-page OCR fallback for scanned / image-only pages
  • Standalone image OCR (pytesseract)
  • Plain-text and Markdown files
  • Text cleaning (OCR noise, encoding artefacts)
  • Sentence-aware chunking with configurable overlap
  • Structured field extraction via Claude (regex fallback if API unavailable)
"""

import io
import logging
import re
from pathlib import Path
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)

# ── Optional dependency guards ────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed — PDF support disabled")

try:
    import pytesseract
    from PIL import Image

    TESSERACT_AVAILABLE = True
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
except (ImportError, Exception):
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract / Pillow not available — OCR disabled")

SUPPORTED_PDF = {".pdf"}
SUPPORTED_IMAGE = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
SUPPORTED_TEXT = {".txt", ".md", ".text", ".rst"}


# ── Low-level extractors ──────────────────────────────────────────────────────

def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Run tesseract over raw image bytes."""
    if not TESSERACT_AVAILABLE:
        return ""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6")


def extract_text_from_image(file_path: str) -> str:
    """OCR a standalone image file."""
    if not TESSERACT_AVAILABLE:
        logger.warning("OCR unavailable — returning empty text for image")
        return ""
    from PIL import Image  # already guarded above

    img = Image.open(file_path).convert("RGB")
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6").strip()


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF.

    Strategy per page:
      1. Attempt native text extraction (fast, exact).
      2. If the page yields < 40 tokens it is likely a scan → OCR at 300 dpi.
    """
    if not PYMUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF not installed — cannot process PDFs")

    doc = fitz.open(file_path)
    pages: list[str] = []

    for page_no, page in enumerate(doc, start=1):
        native = page.get_text("text").strip()
        if len(native.split()) >= 40:
            pages.append(f"--- Page {page_no} ---\n{native}")
        else:
            logger.debug("Page %d has sparse text; applying OCR", page_no)
            pix = page.get_pixmap(dpi=300)
            ocr_text = _ocr_image_bytes(pix.tobytes("png"))
            pages.append(f"--- Page {page_no} (OCR) ---\n{ocr_text}")

    doc.close()
    return "\n\n".join(pages)


def extract_text_from_file(file_path: str) -> tuple[str, str]:
    """
    Route a file to the right extractor.

    Returns (extracted_text, file_type_string).
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in SUPPORTED_PDF:
        return extract_text_from_pdf(file_path), "pdf"
    if suffix in SUPPORTED_IMAGE:
        return extract_text_from_image(file_path), "image"
    if suffix in SUPPORTED_TEXT or suffix == "":
        return path.read_text(errors="replace"), "text"

    # Unknown extension — try reading as text, fallback to empty
    try:
        return path.read_text(errors="replace"), "text"
    except Exception as exc:
        raise ValueError(f"Unsupported file type '{suffix}': {exc}") from exc


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_extracted_text(text: str) -> str:
    """
    Normalise OCR noise and formatting artefacts.

    • Collapses excessive blank lines and spaces
    • Fixes common OCR substitutions (l/1, |/I)
    • Strips null bytes and control characters
    """
    # Remove null bytes / control chars (except newline and tab)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Common OCR substitutions in legal docs
    text = re.sub(r"\b[lI]\b(?=\s*\d)", "1", text)
    text = re.sub(r"(?<=[A-Z])\|(?=[A-Z])", "I", text)
    # Remove stray form-feed characters
    text = text.replace("\x0c", "\n\n")
    return text.strip()


# ── Structured field extraction ───────────────────────────────────────────────

def extract_structured_fields(text: str) -> dict:
    """
    Extract key legal metadata.

    Tries Claude first for high accuracy; falls back to regex if the API
    key is absent or the call fails.
    """
    if settings.ANTHROPIC_API_KEY:
        try:
            return _llm_extract_fields(text)
        except Exception as exc:
            logger.warning("LLM field extraction failed (%s); using regex fallback", exc)
    return _regex_extract_fields(text)


def _llm_extract_fields(text: str) -> dict:
    import json

    from anthropic import Anthropic

    client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    system = (
        "You are a legal document parser. Extract structured metadata from the document.\n"
        "Return ONLY valid JSON matching this schema (use null when a field is absent):\n"
        "{\n"
        '  "case_number": null,\n'
        '  "document_type": null,\n'
        '  "parties": {"plaintiff": null, "defendant": null, "others": []},\n'
        '  "dates": {"filing_date": null, "incident_date": null, "other_dates": []},\n'
        '  "jurisdiction": null,\n'
        '  "attorneys": [],\n'
        '  "key_claims": [],\n'
        '  "contract_value": null,\n'
        '  "keywords": []\n'
        "}"
    )
    resp = client.messages.create(
        model=settings.MODEL,
        max_tokens=800,
        system=system,
        messages=[
            {
                "role": "user",
                "content": f"Extract structured fields from:\n\n{text[:3500]}",
            }
        ],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def _regex_extract_fields(text: str) -> dict:
    """Regex-based fallback for structured field extraction."""
    fields: dict = {
        "case_number": None,
        "document_type": None,
        "parties": {"plaintiff": None, "defendant": None, "others": []},
        "dates": {"filing_date": None, "incident_date": None, "other_dates": []},
        "jurisdiction": None,
        "attorneys": [],
        "key_claims": [],
        "contract_value": None,
        "keywords": [],
    }

    # Case / docket number
    m = re.search(r"(?:Case\s*No\.?|Docket\s*No\.?|No\.)\s*:?\s*([\w\-\/]+)", text, re.I)
    if m:
        fields["case_number"] = m.group(1)

    # Dates (MM/DD/YYYY variants)
    dates = re.findall(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b", text)
    if dates:
        fields["dates"]["other_dates"] = list(dict.fromkeys(dates))[:6]

    # Plaintiff / Defendant
    pm = re.search(r"([A-Z][A-Za-z &,\.]+),?\s+Plaintiff", text)
    if pm:
        fields["parties"]["plaintiff"] = pm.group(1).strip()
    dm = re.search(r"([A-Z][A-Za-z &,\.]+),?\s+Defendant", text)
    if dm:
        fields["parties"]["defendant"] = dm.group(1).strip()

    # Contract / damages value
    vm = re.search(r"\$\s*([\d,]+(?:\.\d{2})?)", text)
    if vm:
        fields["contract_value"] = vm.group(0)

    # Court / jurisdiction
    jm = re.search(
        r"(?:SUPERIOR COURT|DISTRICT COURT|SUPREME COURT|COURT OF [A-Z ]+)\s+(?:OF THE\s+)?(?:STATE OF\s+)?([A-Z][A-Za-z ]+)",
        text,
    )
    if jm:
        fields["jurisdiction"] = jm.group(0).strip()

    # Document type keywords
    for dtype in [
        "COMPLAINT",
        "DEPOSITION",
        "CONTRACT",
        "MOTION",
        "AGREEMENT",
        "DECLARATION",
        "AFFIDAVIT",
        "MEMORANDUM",
        "ORDER",
    ]:
        if re.search(rf"\b{dtype}\b", text, re.I):
            fields["document_type"] = dtype.title()
            break

    return fields


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[str]:
    """
    Split text into overlapping sentence-aware chunks.

    Chunking at sentence boundaries produces more coherent retrieval units
    than naive character slicing. Overlap preserves cross-boundary context.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        s_words = len(sentence.split())

        if current_words + s_words > chunk_size and current:
            chunks.append(" ".join(current))

            # Preserve trailing sentences for overlap
            carry: list[str] = []
            carry_words = 0
            for s in reversed(current):
                w = len(s.split())
                if carry_words + w <= overlap:
                    carry.insert(0, s)
                    carry_words += w
                else:
                    break
            current = carry
            current_words = carry_words

        current.append(sentence)
        current_words += s_words

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if len(c.split()) > 10]


# ── Top-level pipeline ────────────────────────────────────────────────────────

def process_document(file_path: str) -> dict:
    """
    Full document ingestion pipeline.

    Returns a dict with keys:
      raw_text, file_type, structured_fields, chunks, chunk_count
    """
    logger.info("Processing document: %s", file_path)

    raw_text, file_type = extract_text_from_file(file_path)
    clean_text = clean_extracted_text(raw_text)

    if not clean_text:
        raise ValueError("No usable text could be extracted from the document")

    structured_fields = extract_structured_fields(clean_text)
    chunks = chunk_text(clean_text)

    logger.info(
        "Document processed — type=%s, length=%d chars, chunks=%d",
        file_type,
        len(clean_text),
        len(chunks),
    )

    return {
        "raw_text": clean_text,
        "file_type": file_type,
        "structured_fields": structured_fields,
        "chunks": chunks,
        "chunk_count": len(chunks),
    }
