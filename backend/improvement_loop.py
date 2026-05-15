"""
Operator edit → learned pattern pipeline.

When an operator submits an edited draft the system:
  1. Sends the (original, edited) pair to Groq.
  2. Asks the model to distil one reusable improvement pattern from the diff.
  3. Persists the pattern to the LearnedPattern table.
  4. Future calls to generate_draft inject the top patterns into the system prompt.

This is a real improvement loop — not a side-by-side diff store.  The extracted
patterns are semantic descriptions of *what went wrong* and *how to fix it*,
not byte-level diffs.
"""

import json
import logging
import re
from typing import Optional

from sqlalchemy.orm import Session

from backend.config import settings
from backend.models import LearnedPattern, OperatorEdit

logger = logging.getLogger(__name__)


# ── Pattern extraction ────────────────────────────────────────────────────────

def extract_pattern_from_edit(original: str, edited: str, draft_type: str) -> dict:
    """
    Use Groq to distil a reusable improvement pattern from an operator edit.

    Falls back to a simple structural diff if the API is unavailable.
    """
    if settings.GROQ_API_KEYS:
        try:
            return _llm_extract_pattern(original, edited, draft_type)
        except Exception as exc:
            logger.warning("LLM pattern extraction failed (%s); using diff fallback", exc)

    return _diff_pattern(original, edited)


def _llm_extract_pattern(original: str, edited: str, draft_type: str) -> dict:
    from backend.llm_client import get_llm_client

    system = (
        "You analyse differences between an AI-generated legal draft and a human-corrected version.\n"
        "Identify the single most important improvement and return ONLY valid JSON:\n"
        "{\n"
        '  "pattern_description": "Short description of what was wrong and the correct approach",\n'
        '  "pattern_type": "tone|structure|completeness|accuracy|citation|formatting|other",\n'
        '  "relevance_tags": ["tag1", "tag2"],\n'
        '  "example_before": "Representative short snippet from the original draft",\n'
        '  "example_after": "Corresponding improved snippet from the edited version"\n'
        "}\n"
        "Be specific and actionable — the pattern must guide a model to produce better drafts next time."
    )
    resp = get_llm_client().complete(
        max_tokens=600,
        system=system,
        user=(
            f"Draft type: {draft_type}\n\n"
            f"ORIGINAL AI DRAFT (first 2000 chars):\n{original[:2000]}\n\n"
            f"OPERATOR-EDITED VERSION (first 2000 chars):\n{edited[:2000]}\n\n"
            "What is the key improvement pattern?"
        ),
    )
    raw = resp.text.strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def _diff_pattern(original: str, edited: str) -> dict:
    """Fallback: measure structural changes."""
    orig_words = len(original.split())
    edit_words = len(edited.split())
    delta = edit_words - orig_words
    direction = "expanded" if delta > 0 else "condensed"

    return {
        "pattern_description": (
            f"Operator {direction} the draft by {abs(delta)} words "
            f"({orig_words} → {edit_words}). Review content for gaps or verbosity."
        ),
        "pattern_type": "completeness" if delta > 0 else "structure",
        "relevance_tags": [],
        "example_before": original[:200],
        "example_after": edited[:200],
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_edit_and_extract_pattern(
    db: Session,
    draft_id: str,
    original_content: str,
    edited_content: str,
    draft_type: str,
) -> tuple[OperatorEdit, Optional[LearnedPattern]]:
    """
    Persist an operator edit and the pattern distilled from it.

    If an identical pattern description already exists for this draft_type,
    its usage_count is incremented rather than creating a duplicate.

    Returns (OperatorEdit, LearnedPattern | None).
    """
    pattern_data = extract_pattern_from_edit(original_content, edited_content, draft_type)

    edit = OperatorEdit(
        draft_id=draft_id,
        original_content=original_content,
        edited_content=edited_content,
        extracted_pattern=pattern_data,
    )
    db.add(edit)

    # Dedup by description
    description = pattern_data.get("pattern_description", "")
    existing: Optional[LearnedPattern] = (
        db.query(LearnedPattern)
        .filter(
            LearnedPattern.draft_type == draft_type,
            LearnedPattern.pattern_description == description,
        )
        .first()
    )

    if existing:
        existing.usage_count += 1
        db.commit()
        db.refresh(edit)
        return edit, existing

    learned = LearnedPattern(
        draft_type=draft_type,
        pattern_description=description,
        pattern_type=pattern_data.get("pattern_type", "other"),
        example_before=pattern_data.get("example_before", ""),
        example_after=pattern_data.get("example_after", ""),
        relevance_tags=pattern_data.get("relevance_tags", []),
        usage_count=1,
    )
    db.add(learned)
    db.commit()
    db.refresh(edit)
    db.refresh(learned)
    return edit, learned


# ── Pattern retrieval ─────────────────────────────────────────────────────────

def get_relevant_patterns(
    draft_type: str,
    db: Optional[Session],
    limit: int = 5,
) -> list[dict]:
    """
    Return the most-used learned patterns for a draft type.

    These are injected into the system prompt as few-shot examples.
    """
    if db is None:
        return []

    rows = (
        db.query(LearnedPattern)
        .filter(LearnedPattern.draft_type == draft_type)
        .order_by(LearnedPattern.usage_count.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": r.id,
            "pattern_description": r.pattern_description,
            "pattern_type": r.pattern_type,
            "example_before": r.example_before,
            "example_after": r.example_after,
            "relevance_tags": r.relevance_tags,
            "usage_count": r.usage_count,
        }
        for r in rows
    ]
