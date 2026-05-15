"""
Grounded draft generation.

Every draft is produced by:
  1. Formatting retrieved chunks as a numbered evidence block.
  2. Building a system prompt that (a) enforces citation discipline and
     (b) injects any learned patterns as few-shot improvement examples.
  3. Calling Groq with the evidence block + structured metadata.

Citation discipline: the model is instructed to append [Evidence N] after every
factual claim and to write "INSUFFICIENT EVIDENCE" when the evidence does not
support a claim. This keeps all output grounded in source material.
"""

import logging

from backend.config import settings
from backend.llm_client import get_llm_client

logger = logging.getLogger(__name__)

DRAFT_TYPES: dict[str, str] = {
    "case_fact_summary": "Case Fact Summary",
    "title_review": "Title Review Summary",
    "notice_summary": "Notice-Related Summary",
    "document_checklist": "Document Checklist",
    "internal_memo": "First-Pass Internal Memo",
}


# ── Evidence formatting ───────────────────────────────────────────────────────

def format_evidence_block(chunks: list[dict]) -> str:
    """
    Render retrieved chunks as a numbered evidence block.

    Example:
      [Evidence 1] (relevance: 0.91)
      "Defendant agreed to deliver 500 units by March 1, 2024..."
    """
    if not chunks:
        return "⚠  No relevant passages were retrieved for this document."

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        score = chunk.get("relevance_score", 0.0)
        lines.append(f"[Evidence {i}] (relevance: {score:.2f})\n{chunk['text'].strip()}")

    return "\n\n".join(lines)


# ── Prompt construction ───────────────────────────────────────────────────────

def _build_system_prompt(draft_type: str, patterns: list[dict]) -> str:
    label = DRAFT_TYPES.get(draft_type, draft_type)

    prompt = f"""You are a legal document analyst at Pearson Specter Litt.
Your task: produce a grounded {label}.

═══ GROUNDING RULES (non-negotiable) ═══
1. Every factual claim MUST end with an [Evidence N] citation referencing the block above.
2. If the evidence does not support a claim, write "INSUFFICIENT EVIDENCE" — do not fabricate.
3. Do not import knowledge from outside the provided evidence unless clearly labelled "(General legal knowledge)".
4. All monetary figures, dates, party names, and legal conclusions must be directly traceable to a cited evidence block.

═══ STRUCTURE ═══
• Use ## section headers.
• Write concise, professional prose.
• End with a "## Gaps & Uncertainties" section listing anything the evidence did not answer.

═══ OUTPUT FORMAT EXAMPLE ═══
## Parties
ACME Corp is identified as the plaintiff [Evidence 2]. The defendant is GlobalTech LLC [Evidence 1].

## Key Claims
The plaintiff alleges breach of the Master Services Agreement dated January 2024 [Evidence 3].
"""

    if patterns:
        prompt += "\n═══ LEARNED PATTERNS (apply these) ═══\n"
        prompt += "The following improvement patterns were extracted from prior operator edits:\n"
        for i, p in enumerate(patterns[:4], start=1):
            prompt += f"\nPattern {i} ({p.get('pattern_type', 'general')}): {p['pattern_description']}"
            if p.get("example_before") and p.get("example_after"):
                prompt += f"\n  ✗ Before: {p['example_before'][:180]}"
                prompt += f"\n  ✓ After:  {p['example_after'][:180]}"

    return prompt


# ── Main generation function ──────────────────────────────────────────────────

def generate_draft(
    document_id: str,
    doc_metadata: dict,
    retrieved_chunks: list[dict],
    draft_type: str = "case_fact_summary",
    db=None,
) -> dict:
    """
    Generate a grounded draft from retrieved evidence.

    Parameters
    ----------
    document_id    : The source document identifier.
    doc_metadata   : Dict with at least 'structured_fields' and 'filename'.
    retrieved_chunks: Ranked list of chunk dicts from the vector store.
    draft_type     : One of DRAFT_TYPES keys.
    db             : SQLAlchemy session (optional, used to fetch patterns).

    Returns
    -------
    Dict with keys: content, evidence_chunks, prompt_used, patterns_applied
    """
    from backend.improvement_loop import get_relevant_patterns

    patterns = get_relevant_patterns(draft_type, db) if db else []
    system_prompt = _build_system_prompt(draft_type, patterns)
    evidence_block = format_evidence_block(retrieved_chunks)

    # Summarise structured fields for the user turn
    sf = doc_metadata.get("structured_fields") or {}
    fields_summary = (
        f"STRUCTURED METADATA EXTRACTED FROM DOCUMENT:\n"
        f"  Case Number : {sf.get('case_number') or 'Unknown'}\n"
        f"  Doc Type    : {sf.get('document_type') or 'Unknown'}\n"
        f"  Parties     : {sf.get('parties', {})}\n"
        f"  Key Dates   : {sf.get('dates', {})}\n"
        f"  Jurisdiction: {sf.get('jurisdiction') or 'Unknown'}\n"
        f"  Key Claims  : {sf.get('key_claims', [])}\n"
        f"  Value       : {sf.get('contract_value') or 'Not specified'}\n"
    )

    label = DRAFT_TYPES.get(draft_type, draft_type)
    user_msg = (
        f"{fields_summary}\n\n"
        f"RETRIEVED EVIDENCE:\n{evidence_block}\n\n"
        f"Produce a complete {label}. "
        f"Cite [Evidence N] after every factual claim. "
        f"Do not assert anything not supported by the evidence above."
    )

    response = get_llm_client().complete(
        system=system_prompt,
        user=user_msg,
        max_tokens=settings.GROQ_MAX_TOKENS,
    )

    draft_content = response.text

    return {
        "content": draft_content,
        "evidence_chunks": [
            {
                "chunk_index": c.get("chunk_index"),
                "relevance_score": c.get("relevance_score"),
                "text_preview": c["text"][:220],
                "doc_id": c.get("doc_id"),
            }
            for c in retrieved_chunks
        ],
        "prompt_used": user_msg,
        "patterns_applied": len(patterns),
    }
