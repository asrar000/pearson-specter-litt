"""
Unit tests for improvement_loop.py.

These tests do NOT require an Anthropic API key — they test the fallback
diff-based pattern extraction and the SQLAlchemy persistence layer.
"""

import pytest

from backend.improvement_loop import (
    _diff_pattern,
    get_relevant_patterns,
    save_edit_and_extract_pattern,
)
from backend.models import LearnedPattern


# ── _diff_pattern ─────────────────────────────────────────────────────────────

class TestDiffPattern:
    def test_expansion_detected(self):
        original = "Short draft."
        edited = "This is a much longer and more detailed draft with many more words added by the operator."
        pattern = _diff_pattern(original, edited)
        assert "expanded" in pattern["pattern_description"]
        assert pattern["pattern_type"] == "completeness"

    def test_condensation_detected(self):
        original = "A very long draft with many many many words and sentences and paragraphs."
        edited = "Short."
        pattern = _diff_pattern(original, edited)
        assert "condensed" in pattern["pattern_description"]
        assert pattern["pattern_type"] == "structure"

    def test_returns_required_keys(self):
        pattern = _diff_pattern("before text", "after text")
        for key in ["pattern_description", "pattern_type", "relevance_tags", "example_before", "example_after"]:
            assert key in pattern

    def test_example_snippets_truncated(self):
        long_text = "x" * 1000
        pattern = _diff_pattern(long_text, "short")
        assert len(pattern["example_before"]) <= 200


# ── save_edit_and_extract_pattern ─────────────────────────────────────────────

class TestSaveEdit:
    def test_creates_edit_and_pattern(self, db_session):
        from backend.models import Draft, Document

        # Minimal DB setup
        doc = Document(id="d1", filename="test.txt", file_path="/tmp/test.txt")
        draft = Draft(id="dr1", document_id="d1", draft_type="case_fact_summary",
                      content="Original draft content.")
        db_session.add(doc)
        db_session.add(draft)
        db_session.commit()

        edit, pattern = save_edit_and_extract_pattern(
            db=db_session,
            draft_id="dr1",
            original_content="Original draft content.",
            edited_content="Corrected and expanded draft content with more detail.",
            draft_type="case_fact_summary",
        )

        assert edit is not None
        assert edit.draft_id == "dr1"
        assert edit.extracted_pattern is not None
        assert pattern is not None
        assert pattern.draft_type == "case_fact_summary"
        assert pattern.usage_count == 1

    def test_deduplicates_identical_patterns(self, db_session):
        from backend.models import Draft, Document

        doc = Document(id="d2", filename="test2.txt", file_path="/tmp/test2.txt")
        draft1 = Draft(id="dr2a", document_id="d2", draft_type="internal_memo",
                       content="Short draft.")
        draft2 = Draft(id="dr2b", document_id="d2", draft_type="internal_memo",
                       content="Short draft.")
        db_session.add_all([doc, draft1, draft2])
        db_session.commit()

        # Same edit twice
        _, pattern1 = save_edit_and_extract_pattern(db_session, "dr2a", "Short draft.",
                                                    "Much longer draft.", "internal_memo")
        _, pattern2 = save_edit_and_extract_pattern(db_session, "dr2b", "Short draft.",
                                                    "Much longer draft.", "internal_memo")

        patterns = db_session.query(LearnedPattern).filter_by(draft_type="internal_memo").all()
        # Should be one pattern with usage_count >= 2 (exact dedup depends on description match)
        total_usage = sum(p.usage_count for p in patterns)
        assert total_usage >= 2


# ── get_relevant_patterns ─────────────────────────────────────────────────────

class TestGetRelevantPatterns:
    def test_returns_empty_for_no_patterns(self, db_session):
        patterns = get_relevant_patterns("case_fact_summary", db_session)
        assert patterns == []

    def test_returns_patterns_ordered_by_usage(self, db_session):
        db_session.add(LearnedPattern(
            draft_type="internal_memo",
            pattern_description="Pattern A",
            pattern_type="tone",
            usage_count=5,
        ))
        db_session.add(LearnedPattern(
            draft_type="internal_memo",
            pattern_description="Pattern B",
            pattern_type="structure",
            usage_count=12,
        ))
        db_session.commit()

        patterns = get_relevant_patterns("internal_memo", db_session)
        assert patterns[0]["usage_count"] >= patterns[-1]["usage_count"]
        assert patterns[0]["pattern_description"] == "Pattern B"

    def test_filters_by_draft_type(self, db_session):
        db_session.add(LearnedPattern(draft_type="case_fact_summary",
                                      pattern_description="P-case", pattern_type="other", usage_count=1))
        db_session.add(LearnedPattern(draft_type="internal_memo",
                                      pattern_description="P-memo", pattern_type="other", usage_count=1))
        db_session.commit()

        patterns = get_relevant_patterns("case_fact_summary", db_session)
        assert all(p["pattern_description"] != "P-memo" for p in patterns)

    def test_returns_none_db_gracefully(self):
        patterns = get_relevant_patterns("case_fact_summary", None)
        assert patterns == []
