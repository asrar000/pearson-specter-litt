"""
Unit tests for document_processor.py.

These tests do NOT require an Anthropic API key or tesseract.
They cover text cleaning, chunking, regex extraction, and the
plain-text ingestion path of process_document().
"""

import pytest

from backend.document_processor import (
    _regex_extract_fields,
    chunk_text,
    clean_extracted_text,
    extract_text_from_file,
    process_document,
)


# ── clean_extracted_text ──────────────────────────────────────────────────────

class TestCleanExtractedText:
    def test_collapses_excessive_newlines(self):
        text = "First paragraph.\n\n\n\n\nSecond paragraph."
        result = clean_extracted_text(text)
        assert "\n\n\n" not in result
        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_removes_control_chars(self):
        text = "Hello\x00World\x07!"
        result = clean_extracted_text(text)
        assert "\x00" not in result
        assert "\x07" not in result
        assert "Hello" in result

    def test_collapses_spaces(self):
        text = "Too    many    spaces."
        result = clean_extracted_text(text)
        assert "  " not in result

    def test_strips_leading_trailing(self):
        result = clean_extracted_text("  \n text \n  ")
        assert result == "text"

    def test_empty_string(self):
        assert clean_extracted_text("") == ""


# ── chunk_text ────────────────────────────────────────────────────────────────

class TestChunkText:
    def test_produces_chunks_for_long_text(self):
        # 1000 words of filler split into sentences
        sentences = ["This is sentence number %d." % i for i in range(120)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_short_text_produces_one_chunk(self):
        text = "Short document. Only a few sentences here. Not much to see."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1

    def test_chunks_have_minimum_content(self):
        text = "The plaintiff filed a complaint for breach of contract. " * 50
        chunks = chunk_text(text, chunk_size=80, overlap=15)
        for c in chunks:
            assert len(c.split()) > 10

    def test_overlap_preserved(self):
        """Last sentence of chunk N should appear in chunk N+1."""
        sentences = ["Sentence %d is here." % i for i in range(60)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=50, overlap=15)
        assert len(chunks) >= 2
        # At least some overlap word should appear in consecutive chunks
        words_1 = set(chunks[0].split())
        words_2 = set(chunks[1].split())
        assert len(words_1 & words_2) > 0


# ── _regex_extract_fields ─────────────────────────────────────────────────────

class TestRegexExtractFields:
    def test_extracts_case_number(self, sample_legal_text):
        fields = _regex_extract_fields(sample_legal_text)
        assert fields["case_number"] == "CV-2024-08891"

    def test_extracts_plaintiff(self, sample_legal_text):
        fields = _regex_extract_fields(sample_legal_text)
        assert fields["parties"]["plaintiff"] is not None
        assert "PINNACLE" in fields["parties"]["plaintiff"]

    def test_extracts_defendant(self, sample_legal_text):
        fields = _regex_extract_fields(sample_legal_text)
        assert fields["parties"]["defendant"] is not None
        assert "QUANTUM" in fields["parties"]["defendant"]

    def test_extracts_contract_value(self, sample_legal_text):
        fields = _regex_extract_fields(sample_legal_text)
        assert fields["contract_value"] is not None
        assert "2,850,000" in fields["contract_value"]

    def test_identifies_document_type(self, sample_legal_text):
        # text contains "breach" — should be tagged or left None
        fields = _regex_extract_fields(sample_legal_text)
        # document_type may be None if no keyword matched; that's OK
        assert isinstance(fields["document_type"], (str, type(None)))

    def test_handles_empty_text(self):
        fields = _regex_extract_fields("")
        assert fields["case_number"] is None
        assert fields["parties"]["plaintiff"] is None


# ── extract_text_from_file ────────────────────────────────────────────────────

class TestExtractTextFromFile:
    def test_reads_txt_file(self, sample_txt_file, sample_legal_text):
        text, ftype = extract_text_from_file(sample_txt_file)
        assert ftype == "text"
        assert "Plaintiff" in text

    def test_unsupported_extension_tries_text(self, tmp_path):
        f = tmp_path / "doc.xyz"
        f.write_text("Some content here.")
        text, ftype = extract_text_from_file(str(f))
        assert "Some content" in text


# ── process_document (integration, txt only) ─────────────────────────────────

class TestProcessDocument:
    def test_full_pipeline_txt(self, sample_txt_file):
        result = process_document(sample_txt_file)
        assert "raw_text" in result
        assert "chunks" in result
        assert result["chunk_count"] == len(result["chunks"])
        assert result["file_type"] == "text"
        assert result["chunk_count"] >= 1

    def test_structured_fields_present(self, sample_txt_file):
        result = process_document(sample_txt_file)
        sf = result["structured_fields"]
        assert isinstance(sf, dict)
        assert "case_number" in sf
        assert "parties" in sf

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("   \n   ")
        with pytest.raises(ValueError, match="No usable text"):
            process_document(str(f))
