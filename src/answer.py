"""
Grounded natural-language answer generation with citation verification
and date-aware temporal policy versioning.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from google import genai
    from google.genai import types
    HAVE_GEMINI_SDK = True
except ImportError:
    HAVE_GEMINI_SDK = False

try:
    from .models import PolicyClause
    from .loader import load_policy, load_full_policy_corpus
    from .retriever import PolicyRetriever, ScoredClause
    from .temporal import TemporalContext, TemporalStatus, QueryEventType, extract_temporal_context
except ImportError:
    from models import PolicyClause
    from loader import load_policy, load_full_policy_corpus
    from retriever import PolicyRetriever, ScoredClause
    from temporal import TemporalContext, TemporalStatus, QueryEventType, extract_temporal_context


# Default Gemini model and timeout configured for grounded answer generation
DEFAULT_GEMINI_MODEL: str = "gemini-3.6-flash"
DEFAULT_TIMEOUT_SECONDS: float = 60.0


INSUFFICIENT_EVIDENCE_PHRASES = [
    "insufficient evidence",
    "evidence is insufficient",
    "insufficient",
    "does not contain",
    "not contained in the policy",
    "cannot answer",
    "no information",
    "outside the scope",
    "not mentioned in the policy",
    "policy does not specify",
]


@dataclass
class AnswerResult:
    """
    Structured outcome of a grounded answer generation attempt.
    """
    answer: str
    citations: List[str]
    grounded: bool
    insufficient_evidence: bool
    unsupported_citations: List[str] = field(default_factory=list)
    evidence_clause_ids: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None
    temporal_context: Optional[TemporalContext] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "grounded": self.grounded,
            "insufficient_evidence": self.insufficient_evidence,
            "unsupported_citations": self.unsupported_citations,
            "evidence_clause_ids": self.evidence_clause_ids,
            "temporal_status": self.temporal_context.status.value if self.temporal_context else None,
        }

    def __str__(self) -> str:
        status = "GROUNDED" if self.grounded else "UNGROUNDED"
        cits = f"Citations: {self.citations}" if self.citations else "No citations"
        return f"[{status} | {cits}]\n{self.answer}"


def extract_citations(text: str) -> List[str]:
    """
    Extracts all candidate clause IDs (e.g. '4.3.2', '10.5.3A', 'Amendment 2026-01 §5.1') from text.
    """
    seen: Set[str] = set()
    ordered: List[str] = []

    # First extract full Amendment citations e.g. Amendment 2026-01 §5.2 or Amendment No. 2026-01 §5.1
    amend_matches = re.findall(r"(?:Amendment\s+(?:No\.\s+)?2026-01\s+§?(\d+\.\d+))", text, re.IGNORECASE)
    for m in amend_matches:
        full_id = f"Amendment 2026-01 §{m.strip()}"
        if full_id not in seen:
            seen.add(full_id)
            ordered.append(full_id)

    # Remove amendment references before standard clause extraction to prevent false matching on amendment numbers/years
    text_clean = re.sub(r"Amendment\s+(?:No\.\s+)?2026-01\s+§?\d+\.\d+", "", text, flags=re.IGNORECASE)

    # Match standard alphanumeric clauses like §4.3.2, §10.5.3A, 6.4.1
    std_matches = re.findall(r"(?:§|\[§|clause\s+|section\s+)?(\d{1,2}\.\d{1,2}\.\d{1,2}[A-Za-z]?)", text_clean, re.IGNORECASE)
    for m in std_matches:
        clean = m.strip()
        if clean.startswith("2026") or clean.startswith("2025"):
            continue
        if clean and clean not in seen:
            seen.add(clean)
            ordered.append(clean)

    return ordered


def build_grounded_prompt(
    question: str,
    retrieved_clauses: List[Union[PolicyClause, ScoredClause]],
    temporal_context: Optional[TemporalContext] = None,
) -> str:
    """
    Constructs a strict grounding prompt containing only the retrieved evidence and temporal instructions.
    """
    t_ctx = temporal_context or extract_temporal_context(question)

    evidence_blocks: List[str] = []
    for item in retrieved_clauses:
        clause = item.clause if isinstance(item, ScoredClause) else item
        title_str = f" ({clause.clause_title})" if clause.clause_title else ""
        section_str = f" [{clause.parent_section}]" if clause.parent_section else ""
        doc_str = f" [{clause.source_document}]" if clause.source_document else ""
        eff_str = f" [Effective: {clause.effective_date}]" if clause.effective_date else ""
        header = f"[{clause.citation}]{title_str}{section_str}{doc_str}{eff_str}"
        evidence_blocks.append(f"{header}\n{clause.clause_text}")

    evidence_text = "\n\n".join(evidence_blocks)

    temporal_instructions = f"""TEMPORAL APPLICABILITY RULES:
- Detected Query Date: {t_ctx.detected_date_str or 'Unspecified'}
- Temporal Status: {t_ctx.status.value} (Event: {t_ctx.event_type.value})
- Guidance: {t_ctx.explanation}

CRITICAL RULES ON TEMPORAL VALIDITY & AMENDMENTS:
1. If the user question specifies a date before 1 March 2026:
   - Apply the pre-amendment policy rules (e.g. $120 earnings disregard under §6.4.1, 10 calendar days under §4.3.2, 20% sanction under §10.5.2).
2. If the user question specifies a date on or after 1 March 2026:
   - Apply the amended rules under Amendment No. 2026-01 (e.g. $175 earnings disregard under §6.4.1(a), 14 calendar days under §4.3.2, 15% sanction under §10.5.2, or exception under §10.5.3A).
3. Transitional Rules (§5 of Amendment No. 2026-01):
   - §5.1 (Determinations): Amendments to earnings disregard (§6.4.1(a)), income thresholds (§6.6.1), and sanctions (§10.5.2, §10.5.3A) apply to ANY determination made on or after 1 March 2026, even for prior periods.
   - §5.2 (Reporting of Changes): The 14-day reporting period under §4.3.2 and §9.1.4 applies ONLY where the change of circumstances occurred on or after 1 March 2026. If the change occurred before 1 March 2026, the 10-day period applies.
   - §5.3 (Spanning Periods): For claims spanning across 1 March 2026, daily rates apply and the award is apportioned under §7.4.3.
4. If NO specific date is provided in the question:
   - You MUST clearly explain BOTH rules: (1) the rule in force before 1 March 2026, (2) the current rule effective from 1 March 2026 under Amendment No. 2026-01, and (3) cite the applicable transitional provision (e.g. Amendment §5.1 or §5.2)."""

    prompt = f"""You are a helpful, strictly grounded assistant for the Calder County Household Support Program.

CRITICAL INSTRUCTIONS:
1. The supplied POLICY EVIDENCE below is the ONLY authoritative source of information.
2. Answer the USER QUESTION using ONLY the facts explicitly stated in the POLICY EVIDENCE.
3. Do NOT use outside knowledge, general assumptions, or extrapolate beyond the text.
4. If the POLICY EVIDENCE does not contain sufficient facts to answer the question, state clearly: "The provided policy evidence is insufficient to answer this question."
5. For every substantive statement or rule you state, cite the exact clause ID (e.g. §4.3.2, §6.4.1, §10.5.3A, or Amendment 2026-01 §5.2) from the POLICY EVIDENCE that supports it.
6. Do NOT fabricate, guess, or invent clause IDs. Only cite clause IDs that appear in the POLICY EVIDENCE.

{temporal_instructions}

USER QUESTION:
{question}

POLICY EVIDENCE:
{evidence_text}

GROUNDED ANSWER (with exact clause citations):"""
    return prompt


def validate_citations(
    raw_text: str,
    allowed_clause_ids: Set[str],
) -> Tuple[List[str], List[str], bool]:
    """
    Validates all clause citations in the generated response against retrieved evidence.
    """
    found_citations = extract_citations(raw_text)
    valid_citations: List[str] = []
    unsupported_citations: List[str] = []

    # Build normalized allowed lookup tokens
    normalized_allowed: Set[str] = set()
    for cid in allowed_clause_ids:
        normalized_allowed.add(cid.lower())
        clean = cid.lstrip("§").strip().lower()
        normalized_allowed.add(clean)
        if "§" in cid:
            normalized_allowed.add(cid.split("§")[-1].strip().lower())
        if " " in cid:
            for piece in cid.split():
                clean_p = piece.lstrip("§").strip().lower()
                if "." in clean_p:
                    normalized_allowed.add(clean_p)

    for cid in found_citations:
        clean = cid.lstrip("§").strip().lower()
        if (
            cid.lower() in normalized_allowed
            or clean in normalized_allowed
            or any(clean == a.lstrip("§").strip().lower() for a in allowed_clause_ids)
            or any(clean in a.lower() for a in allowed_clause_ids)
        ):
            valid_citations.append(cid)
        else:
            unsupported_citations.append(cid)

    # Grounded if no unsupported citations were fabricated
    is_grounded = len(unsupported_citations) == 0
    return valid_citations, unsupported_citations, is_grounded


def check_insufficient_evidence(text: str) -> bool:
    """Detects whether the answer states that evidence is insufficient."""
    lower = text.lower()
    return any(phrase in lower for phrase in INSUFFICIENT_EVIDENCE_PHRASES)


def generate_answer(
    question: str,
    retrieved_clauses: List[Union[PolicyClause, ScoredClause]],
    client: Optional[Any] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    temporal_context: Optional[TemporalContext] = None,
) -> AnswerResult:
    """
    Generates a grounded natural-language answer using Gemini and retrieved policy evidence.
    
    Args:
        question: The user inquiry.
        retrieved_clauses: List of retrieved PolicyClause or ScoredClause objects.
        client: Optional pre-configured Gemini client or mock client.
        model: Gemini model identifier (defaults to GEMINI_MODEL env var or 'gemini-3.6-flash').
        api_key: Optional Gemini API key override (defaults to GEMINI_API_KEY env var).
        temperature: Generation temperature (default 0.0 for deterministic grounding).
        timeout_seconds: API request timeout in seconds (default 60.0s).
        temporal_context: Optional pre-computed TemporalContext.
        
    Returns:
        AnswerResult containing answer text, validated citations, and grounding status.
    """
    t_ctx = temporal_context or extract_temporal_context(question)

    # Extract allowed clause IDs
    allowed_ids: Set[str] = set()
    for item in retrieved_clauses:
        clause = item.clause if isinstance(item, ScoredClause) else item
        allowed_ids.add(clause.clause_id)
        if clause.citation:
            allowed_ids.add(clause.citation)
        if clause.hierarchy.get("amendment_paragraph"):
            allowed_ids.add(clause.hierarchy["amendment_paragraph"])
        if clause.transitional_rule:
            allowed_ids.add(f"Amendment 2026-01 §{clause.transitional_rule}")
            allowed_ids.add(f"§{clause.transitional_rule}")
            allowed_ids.add(clause.transitional_rule)

    # Fast-path: If no evidence was retrieved, refuse without calling LLM
    if not retrieved_clauses or not allowed_ids:
        return AnswerResult(
            answer="The provided policy evidence is insufficient to answer this question.",
            citations=[],
            grounded=True,
            insufficient_evidence=True,
            evidence_clause_ids=[],
            raw_response=None,
            temporal_context=t_ctx,
        )

    prompt = build_grounded_prompt(question, retrieved_clauses, temporal_context=t_ctx)
    effective_model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    http_opts = types.HttpOptions(timeout=int(timeout_seconds * 1000)) if HAVE_GEMINI_SDK else None

    # Handle Gemini invocation or mock client with graceful exception handling
    try:
        if client is not None:
            # Mock or custom client
            if hasattr(client, "generate"):
                raw_response = client.generate(prompt)
            elif hasattr(client, "models") and hasattr(client.models, "generate_content"):
                res = client.models.generate_content(
                    model=effective_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        http_options=http_opts,
                    ) if HAVE_GEMINI_SDK else None,
                )
                raw_response = res.text if hasattr(res, "text") else str(res)
            elif callable(client):
                raw_response = client(prompt)
            else:
                raise ValueError(f"Unsupported client object type: {type(client)}")
        else:
            # Live Gemini API call
            resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "Gemini API key is not configured. Please set the GEMINI_API_KEY environment variable "
                    "or pass an explicit api_key or mock client."
                )

            if not HAVE_GEMINI_SDK:
                raise ImportError(
                    "google-genai SDK is not installed. Please install google-genai to use live API calls."
                )

            live_client = genai.Client(api_key=resolved_key, http_options=http_opts)
            res = live_client.models.generate_content(
                model=effective_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    http_options=http_opts,
                ),
            )
            raw_response = res.text or ""
    except Exception as err:
        return AnswerResult(
            answer=f"Error communicating with Gemini API: {err}",
            citations=[],
            grounded=False,
            insufficient_evidence=True,
            unsupported_citations=[],
            evidence_clause_ids=sorted(list(allowed_ids)),
            raw_response=None,
            temporal_context=t_ctx,
        )

    # Validate citations and analyze grounding
    valid_citations, unsupported_citations, is_grounded = validate_citations(raw_response, allowed_ids)
    is_insufficient = check_insufficient_evidence(raw_response)

    return AnswerResult(
        answer=raw_response.strip(),
        citations=valid_citations,
        grounded=is_grounded,
        insufficient_evidence=is_insufficient,
        unsupported_citations=unsupported_citations,
        evidence_clause_ids=sorted(list(allowed_ids)),
        raw_response=raw_response,
        temporal_context=t_ctx,
    )


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("Grounded Answer Generation Demonstration (with Temporal Versioning)")
    print("=" * 70)

    retriever = PolicyRetriever()

    demo_questions = [
        "What is the deadline for reporting a change of circumstances that occurred on 10 February 2026?",
        "What is the deadline for reporting a change of circumstances that occurred on 15 April 2026?",
        "What is the earnings disregard for a determination made on 15 March 2026?",
        "A change occurred on 20 February 2026 and the determination was made on 20 March 2026. What reporting deadline applies?",
        "The claim was from January 2026 but the determination was made on 20 March 2026. What earnings disregard applies?",
        "What is the deadline for reporting a change of circumstances?",
    ]

    for demo_question in demo_questions:
        print(f"\nQuestion: {demo_question}\n")
        retrieved = retriever.retrieve(demo_question, top_k=5)
        print("Retrieved Clause IDs:", [f"{r.citation}" for r in retrieved])

        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            print("Using live Gemini API...")
            result = generate_answer(demo_question, retrieved, api_key=api_key)
        else:
            # Deterministic mock simulation for offline verification
            def mock_llm(p: str) -> str:
                if "20 february 2026" in p.lower() and "20 march 2026" in p.lower():
                    return (
                        "Under Amendment No. 2026-01 §5.2, because the change of circumstances occurred on 20 February 2026 "
                        "(before 1 March 2026), the pre-amendment reporting deadline applies regardless of the determination date. "
                        "Under §4.3.2, the recipient must report the change within **10 calendar days**."
                    )
                elif "january 2026" in p.lower() and "20 march 2026" in p.lower():
                    return (
                        "Under Amendment No. 2026-01 §5.1, amendments apply to any determination made on or after 1 March 2026, "
                        "even for a prior period such as January 2026. Therefore, under §6.4.1 as amended, the earnings disregard is **$175 per month**."
                    )
                elif "10 february 2026" in p.lower():
                    return (
                        "For a change of circumstances occurring on 10 February 2026, the pre-amendment deadline applies under "
                        "Amendment No. 2026-01 §5.2. Under §4.3.2, the recipient must report the change within **10 calendar days**."
                    )
                elif "15 april 2026" in p.lower():
                    return (
                        "For a change of circumstances occurring on 15 April 2026, the amended deadline applies under "
                        "Amendment No. 2026-01 §2.1 and §5.2. Under §4.3.2 as amended, the recipient must report within **14 calendar days**."
                    )
                elif "15 march 2026" in p.lower():
                    return (
                        "For a determination made on 15 March 2026, the amended earnings disregard applies under "
                        "Amendment No. 2026-01 §1.1 and §5.1. Under §6.4.1(a), the first **$175 per month** of earnings is disregarded."
                    )
                else:
                    return (
                        "Prior to 1 March 2026, a recipient must report changes within **10 calendar days** under §4.3.2. "
                        "Effective 1 March 2026 under Amendment No. 2026-01 §2.1, the reporting period is **14 calendar days**. "
                        "Under Amendment No. 2026-01 §5.2, the 14-day rule applies only to changes occurring on or after 1 March 2026."
                    )

            result = generate_answer(demo_question, retrieved, client=mock_llm)

        print("\nGenerated Answer:")
        print(result.answer)
        print(f"Validated Citations: {result.citations}")
        print(f"Grounded Status: {result.grounded}")
        print(f"Temporal Status: {result.temporal_context.status.value if result.temporal_context else None}")
        print("-" * 70)
