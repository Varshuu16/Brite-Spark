"""
Grounded natural-language answer generation with citation verification,
evidence traceability, citation completeness analysis, temporal policy versioning,
and deterministic substantive conflict detection.
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
    from .models import PolicyClause, PolicyCitation, PolicyConflict
    from .loader import load_policy, load_full_policy_corpus
    from .retriever import PolicyRetriever, ScoredClause
    from .temporal import TemporalContext, TemporalStatus, QueryEventType, extract_temporal_context
    from .conflict import detect_conflicts
except ImportError:
    from models import PolicyClause, PolicyCitation, PolicyConflict
    from loader import load_policy, load_full_policy_corpus
    from retriever import PolicyRetriever, ScoredClause
    from temporal import TemporalContext, TemporalStatus, QueryEventType, extract_temporal_context
    from conflict import detect_conflicts


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
    "unsupported by the provided evidence",
    "no evidence provided",
    "does not provide information",
    "policy evidence does not",
]


@dataclass
class AnswerResult:
    """
    Structured outcome of a grounded answer generation attempt with full citation traceability
    and substantive conflict reporting.
    """
    answer: str
    citations: List[str] = field(default_factory=list)
    validated_citations: List[PolicyCitation] = field(default_factory=list)
    unsupported_citations: List[str] = field(default_factory=list)
    grounded: bool = True
    citation_complete: bool = True
    has_missing_citations: bool = False
    insufficient_evidence: bool = False
    conflicts_detected: bool = False
    conflicts: List[PolicyConflict] = field(default_factory=list)
    conflict_status: str = "NO_CONFLICT"
    evidence_clause_ids: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None
    temporal_context: Optional[TemporalContext] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "validated_citations": [vc.to_dict() for vc in self.validated_citations],
            "unsupported_citations": self.unsupported_citations,
            "grounded": self.grounded,
            "citation_complete": self.citation_complete,
            "has_missing_citations": self.has_missing_citations,
            "insufficient_evidence": self.insufficient_evidence,
            "conflicts_detected": self.conflicts_detected,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "conflict_status": self.conflict_status,
            "evidence_clause_ids": self.evidence_clause_ids,
            "temporal_status": self.temporal_context.status.value if self.temporal_context else None,
        }

    def __str__(self) -> str:
        status = "GROUNDED" if self.grounded else "UNGROUNDED"
        complete_str = "COMPLETE" if self.citation_complete else "MISSING CITATIONS"
        conf_str = "CONFLICT" if self.conflicts_detected else "NO_CONFLICT"
        cits = f"Citations: {self.citations}" if self.citations else "No citations"
        return f"[{status} | {conf_str} | {complete_str} | {cits}]\n{self.answer}"


def extract_citations(text: str) -> List[str]:
    """
    Extracts all candidate clause IDs (e.g. '§4.3.2', '§10.5.3A', 'Amendment 2026-01 §5.1') from text.
    Strictly filters out ordinary numbers, currency amounts, and dates.
    """
    seen: Set[str] = set()
    ordered: List[str] = []

    # 1. Extract Amendment citations, e.g. [Amendment 2026-01 §5.2] or Amendment No. 2026-01 §5.1
    amend_matches = re.findall(
        r"\[?(?:Amendment\s+(?:No\.\s+)?2026-01\s+§?(\d+\.\d+))\]?",
        text,
        re.IGNORECASE
    )
    for m in amend_matches:
        full_id = f"Amendment 2026-01 §{m.strip()}"
        if full_id not in seen:
            seen.add(full_id)
            ordered.append(full_id)

    # 2. Remove amendment substrings to prevent false matches on amendment year or paragraph numbers
    text_clean = re.sub(
        r"\[?Amendment\s+(?:No\.\s+)?2026-01\s+§?\d+\.\d+\]?",
        "",
        text,
        flags=re.IGNORECASE
    )

    # 3. Match standard clauses with mandatory section/clause prefix:
    # Requires '§', '[§', 'clause ', or 'section ' before clause identifier
    # Matches: §4.3.2, [§4.3.2], §10.5.3A, section 6.4.1, clause 1.4.6, §6.4.1(a)
    std_matches = re.findall(
        r"(?:§|\[§|clause\s+|section\s+)(\d{1,2}\.\d{1,2}(?:\.\d{1,2}[A-Za-z]?(?:\([a-z]\))?)?)",
        text_clean,
        re.IGNORECASE
    )
    for m in std_matches:
        clean = m.strip()
        # Avoid year numbers or malformed tokens
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
    conflicts: Optional[List[PolicyConflict]] = None,
) -> str:
    """
    Constructs a strict grounding prompt containing retrieved evidence, temporal instructions,
    and conflict notices where applicable.
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

    conflict_instructions = ""
    if conflicts:
        conflict_descriptions = "\n".join(f"- {c.description}" for c in conflicts)
        conflict_instructions = f"""
POLICY CONFLICT NOTICE:
The retrieved policy evidence contains substantive discrepancies or conflicting provisions:
{conflict_descriptions}

CRITICAL CONFLICT REQUIREMENT:
If the retrieved evidence contains conflicting substantive rules, you MUST explicitly state that the evidence contains conflicting provisions, cite BOTH conflicting clauses, and NOT arbitrarily choose one rule over the other."""

    prompt = f"""You are a helpful, strictly grounded assistant for the Calder County Household Support Program.

CRITICAL INSTRUCTIONS:
1. The supplied POLICY EVIDENCE below is the ONLY authoritative source of information.
2. Answer the USER QUESTION using ONLY the facts explicitly stated in the POLICY EVIDENCE.
3. Do NOT use outside knowledge, general assumptions, or extrapolate beyond the text.
4. If the POLICY EVIDENCE does not contain sufficient facts to answer the question, state clearly: "The provided policy evidence is insufficient to answer this question."
5. For every substantive statement or rule you state, cite the exact clause ID (e.g. §4.3.2, §6.4.1, §10.5.3A, or Amendment 2026-01 §5.2) from the POLICY EVIDENCE that supports it.
6. Do NOT fabricate, guess, or invent clause IDs. Only cite clause IDs that appear in the POLICY EVIDENCE.

{temporal_instructions}
{conflict_instructions}

USER QUESTION:
{question}

POLICY EVIDENCE:
{evidence_text}

GROUNDED ANSWER (with exact clause citations):"""
    return prompt


def validate_citations(
    raw_text: str,
    allowed_evidence: Union[Set[str], List[Union[PolicyClause, ScoredClause]]],
) -> Tuple[List[str], List[str], bool]:
    """
    Validates all clause citations in the generated response against retrieved evidence.
    
    Args:
        raw_text: The generated answer text containing citations.
        allowed_evidence: Set of allowed clause ID strings OR list of retrieved clause objects.
        
    Returns:
        Tuple of (valid_citations, unsupported_citations, is_grounded)
    """
    found_citations = extract_citations(raw_text)
    valid_citations: List[str] = []
    unsupported_citations: List[str] = []

    # Build allowed token lookup set and clause object map
    normalized_allowed: Set[str] = set()
    clause_objects_by_token: Dict[str, PolicyClause] = {}

    if isinstance(allowed_evidence, set):
        for cid in allowed_evidence:
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
    else:
        for item in allowed_evidence:
            clause = item.clause if isinstance(item, ScoredClause) else item
            tokens_for_clause = [
                clause.clause_id.lower(),
                clause.clause_id.lstrip("§").strip().lower(),
                clause.citation.lower(),
                clause.citation.lstrip("§").strip().lower(),
            ]
            if clause.hierarchy.get("amendment_paragraph"):
                tokens_for_clause.append(clause.hierarchy["amendment_paragraph"].lower())
                tokens_for_clause.append(clause.hierarchy["amendment_paragraph"].lstrip("§").strip().lower())
            if clause.transitional_rule:
                tokens_for_clause.append(f"amendment 2026-01 §{clause.transitional_rule}".lower())
                tokens_for_clause.append(f"§{clause.transitional_rule}".lower())
                tokens_for_clause.append(clause.transitional_rule.lower())

            for tok in tokens_for_clause:
                normalized_allowed.add(tok)
                clause_objects_by_token[tok] = clause

    for cid in found_citations:
        clean = cid.lstrip("§").strip().lower()
        clean_base = re.sub(r"\([a-z]\)", "", clean).strip()
        if (
            cid.lower() in normalized_allowed
            or clean in normalized_allowed
            or clean_base in normalized_allowed
            or any(clean == a.lstrip("§").strip().lower() for a in normalized_allowed)
            or any(clean in a.lower() for a in normalized_allowed)
        ):
            valid_citations.append(cid)
        else:
            unsupported_citations.append(cid)

    is_grounded = len(unsupported_citations) == 0
    return valid_citations, unsupported_citations, is_grounded


def validate_citations_detailed(
    raw_text: str,
    retrieved_clauses: List[Union[PolicyClause, ScoredClause]],
) -> Tuple[List[str], List[PolicyCitation], List[str], bool]:
    """
    Validates all clause citations in the generated response against retrieved evidence,
    returning structured PolicyCitation objects linked directly to the evidence.
    """
    found_citations = extract_citations(raw_text)
    valid_citation_ids: List[str] = []
    validated_objects: List[PolicyCitation] = []
    unsupported_citations: List[str] = []

    # Map retrieved clauses by all their normalized identifiers
    clause_map: Dict[str, PolicyClause] = {}
    for item in retrieved_clauses:
        clause = item.clause if isinstance(item, ScoredClause) else item
        cid_lower = clause.clause_id.lower()
        clause_map[cid_lower] = clause
        clause_map[cid_lower.lstrip("§").strip()] = clause

        cit_lower = clause.citation.lower()
        clause_map[cit_lower] = clause
        clause_map[cit_lower.lstrip("§").strip()] = clause

        if clause.hierarchy.get("amendment_paragraph"):
            ap_lower = clause.hierarchy["amendment_paragraph"].lower()
            clause_map[ap_lower] = clause
            clause_map[ap_lower.lstrip("§").strip()] = clause
            clause_map[re.sub(r"§", "", ap_lower).strip()] = clause

        if clause.transitional_rule:
            t_rule = clause.transitional_rule.lower()
            clause_map[f"amendment 2026-01 §{t_rule}"] = clause
            clause_map[f"amendment 2026-01 {t_rule}"] = clause
            clause_map[f"§{t_rule}"] = clause
            clause_map[t_rule] = clause

    for cid in found_citations:
        cid_norm = cid.strip().lower()
        clean = cid.lstrip("§").strip().lower()
        clean_no_sec = re.sub(r"§", "", cid_norm).strip()
        clean_base = re.sub(r"\([a-z]\)", "", clean).strip()

        matched_clause: Optional[PolicyClause] = None

        if cid_norm in clause_map:
            matched_clause = clause_map[cid_norm]
        elif clean in clause_map:
            matched_clause = clause_map[clean]
        elif clean_no_sec in clause_map:
            matched_clause = clause_map[clean_no_sec]
        elif clean_base in clause_map:
            matched_clause = clause_map[clean_base]

        if matched_clause is not None:
            valid_citation_ids.append(cid)
            canonical_id = f"§{cid}" if not cid.startswith("§") and not cid.startswith("Amendment") else cid
            validated_objects.append(
                PolicyCitation(
                    citation_id=canonical_id,
                    source_document=matched_clause.source_document,
                    clause_id=matched_clause.clause_id,
                    clause_title=matched_clause.clause_title,
                    clause_text=matched_clause.clause_text[:200] if matched_clause.clause_text else None,
                    is_amendment=matched_clause.is_amendment,
                    is_transitional=matched_clause.is_transitional,
                    amended_by=matched_clause.amended_by,
                    transitional_rule=matched_clause.transitional_rule,
                )
            )
        else:
            unsupported_citations.append(cid)

    is_grounded = len(unsupported_citations) == 0
    return valid_citation_ids, validated_objects, unsupported_citations, is_grounded


def check_insufficient_evidence(text: str) -> bool:
    """Detects whether the answer states that evidence is insufficient."""
    lower = text.lower()
    return any(phrase in lower for phrase in INSUFFICIENT_EVIDENCE_PHRASES)


def check_citation_completeness(
    answer: str,
    valid_citations: List[Any],
    insufficient_evidence: bool = False,
) -> bool:
    """
    Deterministically verifies whether substantive policy claims in the answer are supported by citations.
    
    Returns:
        True if the answer has valid citations OR is a refusal / uncertainty statement.
        False if the answer makes factual policy claims but fails to provide citations.
    """
    # If the response indicates insufficient evidence / refusal, no policy citation is required
    if insufficient_evidence or check_insufficient_evidence(answer):
        return True

    # If valid citations are present, citation completeness is satisfied
    if len(valid_citations) > 0:
        return True

    # When no citations are present, check if the response makes substantive policy assertions
    lower = answer.lower()

    # Patterns indicating substantive administrative policy claims
    policy_patterns = [
        r"\b\d+\s+(?:calendar\s+)?(?:days|weeks|months|years)\b",  # e.g., 10 calendar days, 4 weeks
        r"\$\d+",                                                    # e.g., $120, $175
        r"\b\d+\s*(?:%|percent|per cent)\b",                        # e.g., 15 per cent, 20%
        r"\b(?:must\s+report|required\s+to|obligation|entitled\s+to|eligible|ineligible)\b",
        r"\b(?:disregard\s+is|disregarded|threshold\s+is|sanction\s+is|reduction\s+of)\b",
        r"\b(?:deadline\s+is|period\s+is|within\s+\d+|apportioned)\b",
    ]

    has_substantive_claims = any(re.search(pat, lower) for pat in policy_patterns)

    # If substantive claims were asserted without any citation, completeness failed
    if has_substantive_claims:
        return False

    return True


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
    Generates a grounded natural-language answer using Gemini, retrieved policy evidence,
    rigorous citation verification, and deterministic conflict detection.
    
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
        AnswerResult containing answer text, validated citations, conflict metadata, and grounding status.
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
            validated_citations=[],
            grounded=False,
            citation_complete=True,
            has_missing_citations=False,
            insufficient_evidence=True,
            conflicts_detected=False,
            conflicts=[],
            conflict_status="NO_CONFLICT",
            evidence_clause_ids=[],
            raw_response=None,
            temporal_context=t_ctx,
        )

    # Detect substantive conflicts across retrieved evidence
    detected_conflicts = detect_conflicts(retrieved_clauses, temporal_context=t_ctx, question=question)
    has_conflicts = len(detected_conflicts) > 0

    prompt = build_grounded_prompt(
        question,
        retrieved_clauses,
        temporal_context=t_ctx,
        conflicts=detected_conflicts if has_conflicts else None,
    )
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
            validated_citations=[],
            grounded=False,
            citation_complete=True,
            has_missing_citations=False,
            insufficient_evidence=True,
            conflicts_detected=has_conflicts,
            conflicts=detected_conflicts,
            conflict_status="CONFLICT_DETECTED" if has_conflicts else "NO_CONFLICT",
            unsupported_citations=[],
            evidence_clause_ids=sorted(list(allowed_ids)),
            raw_response=None,
            temporal_context=t_ctx,
        )

    # Validate citations and analyze grounding and completeness
    valid_citations, validated_objects, unsupported_citations, is_grounded = validate_citations_detailed(
        raw_response, retrieved_clauses
    )
    is_insufficient = check_insufficient_evidence(raw_response) and len(valid_citations) == 0
    if is_insufficient:
        valid_citations = []
        validated_objects = []
        unsupported_citations = []
        is_grounded = False
        is_citation_complete = True
    else:
        is_citation_complete = check_citation_completeness(raw_response, valid_citations, is_insufficient)

    return AnswerResult(
        answer=raw_response.strip(),
        citations=valid_citations,
        validated_citations=validated_objects,
        grounded=is_grounded,
        citation_complete=is_citation_complete,
        has_missing_citations=not is_citation_complete,
        insufficient_evidence=is_insufficient,
        conflicts_detected=has_conflicts,
        conflicts=detected_conflicts,
        conflict_status="CONFLICT_DETECTED" if has_conflicts else "NO_CONFLICT",
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
    print("Grounded Answer Generation Demonstration (with Conflict Handling)")
    print("=" * 70)

    retriever = PolicyRetriever()

    demo_scenarios = [
        ("What is the earnings disregard for a determination on 15 March 2026?", None),
        ("What is the deadline for reporting a change of circumstances?", None),
        ("What is the capital of France?", None),
    ]

    for demo_question, forced_evidence in demo_scenarios:
        print(f"\nQuestion: {demo_question}\n")
        if forced_evidence is not None:
            retrieved = forced_evidence
            print("Supplied (Forced) Evidence IDs:", [f"{r.citation}" for r in retrieved])
        else:
            retrieved = retriever.retrieve(demo_question, top_k=5)
            print("Retrieved Clause IDs:", [f"{r.citation}" for r in retrieved])

        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            print("Using live Gemini API...")
            result = generate_answer(demo_question, retrieved, api_key=api_key)
        else:
            # Deterministic mock simulation for offline verification
            def mock_llm(p: str) -> str:
                q_part = p
                if "USER QUESTION:" in p:
                    q_part = p.split("USER QUESTION:")[-1].split("POLICY EVIDENCE:")[0]
                q_low = q_part.lower()

                evidence_text = p.split("POLICY EVIDENCE:")[-1].lower() if "POLICY EVIDENCE:" in p else p.lower()
                if "france" in q_low or "weather" in q_low or "property tax" in q_low:
                    return "The provided policy evidence is insufficient to answer this question."
                elif "15 march 2026" in q_low:
                    return (
                        "For a determination made on 15 March 2026, the amended earnings disregard applies under "
                        "Amendment No. 2026-01 §1.1 and §5.1. Under §6.4.1(a), the first **$175 per month** of earnings is disregarded."
                    )
                elif "change of circumstances" in q_low:
                    return (
                        "The retrieved policy evidence contains two related references with different timeframes: "
                        "under §4.3.2, the recipient's direct obligation is to report within **10 calendar days** (or 14 calendar days on/after 1 March 2026 under Amendment No. 2026-01 §2.1 and §5.2), "
                        "whereas §9.1.4 references a **30 calendar day** period regarding overpayment establishment notifications."
                    )
                else:
                    return (
                        "Prior to 1 March 2026, a recipient must report changes within **10 calendar days** under §4.3.2. "
                        "Effective 1 March 2026 under Amendment No. 2026-01 §2.1, the reporting period is **14 calendar days**. "
                        "Under Amendment No. 2026-01 §5.2, the 14-day rule applies only to changes occurring on or after 1 March 2026."
                    )

            result = generate_answer(demo_question, retrieved, client=mock_llm)

        extracted_cits = extract_citations(result.answer)
        print("\nGenerated Answer:")
        print(result.answer)
        print(f"Extracted Citations: {extracted_cits}")
        print(f"Validated Citations: {result.citations}")
        print(f"Validated Citation Objects: {[str(vc) for vc in result.validated_citations]}")
        print(f"Unsupported Citations: {result.unsupported_citations}")
        print(f"Conflict Detected: {result.conflicts_detected}")
        if result.conflicts_detected:
            print(f"Detected Conflicts: {[c.description for c in result.conflicts]}")
        print(f"Citation Completeness: {result.citation_complete}")
        print(f"Grounded Status: {result.grounded}")
        print(f"Insufficient Evidence: {result.insufficient_evidence}")
        print(f"Temporal Status: {result.temporal_context.status.value if result.temporal_context else None}")
        print("-" * 70)
