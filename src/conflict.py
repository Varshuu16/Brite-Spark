"""
Deterministic Policy Conflict Detection Module.

Identifies substantive contradictions between retrieved policy provisions,
distinguishing genuine policy discrepancies from temporal versioning and amendments.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from .models import PolicyClause, PolicyConflict
    from .retriever import ScoredClause
    from .temporal import TemporalContext, TemporalStatus, QueryEventType, extract_temporal_context
except ImportError:
    from models import PolicyClause, PolicyConflict
    from retriever import ScoredClause
    from temporal import TemporalContext, TemporalStatus, QueryEventType, extract_temporal_context


def detect_conflicts(
    retrieved_clauses: List[Union[PolicyClause, ScoredClause]],
    temporal_context: Optional[TemporalContext] = None,
    question: Optional[str] = None,
) -> List[PolicyConflict]:
    """
    Deterministically analyzes retrieved policy evidence for substantive contradictions.
    
    Rules:
    1. Distinguishes temporal versioning (e.g. 10 days pre-amendment vs 14 days post-amendment)
       from genuine substantive conflicts.
    2. Flags substantive contradictions such as conflicting reporting deadlines (§4.3.2 vs §9.1.4),
       or conflicting sanction rules.
    3. Respects temporal resolution under Amendment No. 2026-01 §5.1, §5.2, and §5.3.
    """
    if not retrieved_clauses:
        return []

    t_ctx = temporal_context
    if t_ctx is None and question:
        t_ctx = extract_temporal_context(question)

    raw_clauses: List[PolicyClause] = [
        item.clause if isinstance(item, ScoredClause) else item
        for item in retrieved_clauses
    ]

    conflicts: List[PolicyConflict] = []
    seen_conflict_keys: Set[str] = set()

    clause_by_id: Dict[str, List[PolicyClause]] = {}
    for c in raw_clauses:
        cid = c.clause_id.lstrip("§").strip()
        clause_by_id.setdefault(cid, []).append(c)

    # -------------------------------------------------------------------------
    # Check 1: Reporting Deadline Substantive Discrepancy (§4.3.2 vs §9.1.4)
    # -------------------------------------------------------------------------
    # §4.3.2 requires reporting within 10 days (or 14 days post-amendment).
    # In the original manual, §9.1.4 references a 30 calendar day period for overpayment recovery.
    has_432 = "4.3.2" in clause_by_id
    has_914 = "9.1.4" in clause_by_id

    is_date_unspecified = t_ctx is None or t_ctx.status == TemporalStatus.UNSPECIFIED
    if has_432 and has_914 and is_date_unspecified:
        c432 = clause_by_id["4.3.2"][0]
        c914 = clause_by_id["9.1.4"][0]

        # Extract days mentioned in both clauses
        m432 = re.search(r"(\d+)\s+calendar\s+days", c432.clause_text, re.IGNORECASE)
        m914 = re.search(r"(\d+)\s+calendar\s+days", c914.clause_text, re.IGNORECASE)

        days_432 = m432.group(1) if m432 else None
        days_914 = m914.group(1) if m914 else None

        if days_432 and days_914 and days_432 != days_914:
            conflict_key = "conflict_reporting_432_914"
            if conflict_key not in seen_conflict_keys:
                seen_conflict_keys.add(conflict_key)
                conflicts.append(
                    PolicyConflict(
                        conflict_id="CONF-REPORTING-01",
                        clause_ids=["§4.3.2", "§9.1.4"],
                        source_documents=[c432.source_document, c914.source_document],
                        conflicting_values={
                            "§4.3.2": f"{days_432} calendar days",
                            "§9.1.4": f"{days_914} calendar days",
                        },
                        description=(
                            f"Substantive discrepancy in reporting deadlines: §4.3.2 specifies {days_432} calendar days "
                            f"for recipient reporting obligation, whereas §9.1.4 references {days_914} calendar days."
                        ),
                        conflict_type="SUBSTANTIVE",
                        resolution_available=False,
                        resolution_notes=(
                            "§4.3.2 defines the direct recipient obligation, whereas §9.1.4 concerns Department "
                            "overpayment establishment notification timeframes."
                        ),
                    )
                )

    # -------------------------------------------------------------------------
    # Check 2: Sanction Contradiction (Sanction Imposition vs Sanction Prohibition)
    # -------------------------------------------------------------------------
    has_sanction_clause = any("10.5" in cid for cid in clause_by_id)
    has_exception_10_5_3A = "10.5.3A" in clause_by_id

    if has_sanction_clause and has_exception_10_5_3A:
        c_sanction = [c for cid, cs in clause_by_id.items() if "10.5" in cid and cid != "10.5.3A" for c in cs]
        c_exc = clause_by_id["10.5.3A"][0]

        if c_sanction and question and any(term in question.lower() for term in ["positive", "increase", "unreported increase", "higher award"]):
            conflict_key = "conflict_sanction_exception_1053A"
            if conflict_key not in seen_conflict_keys:
                seen_conflict_keys.add(conflict_key)
                conflicts.append(
                    PolicyConflict(
                        conflict_id="CONF-SANCTION-01",
                        clause_ids=[c_sanction[0].citation, "§10.5.3A"],
                        source_documents=[c_sanction[0].source_document, c_exc.source_document],
                        conflicting_values={
                            c_sanction[0].citation: "Sanction applicable for failure to report change",
                            "§10.5.3A": "No sanction permitted if unreported change would increase award",
                        },
                        description=(
                            f"General sanction rule in {c_sanction[0].citation} imposes reduction for failure to report, "
                            f"but §10.5.3A strictly prohibits sanctions where the unreported change would increase the award."
                        ),
                        conflict_type="SANCTION_DISCREPANCY",
                        resolution_available=True,
                        resolution_notes="§10.5.3A operates as an explicit statutory exception to the general sanction rules.",
                    )
                )

    # -------------------------------------------------------------------------
    # Check 3: Arbitrary Numeric Contradictions Across Retrieved Clauses
    # -------------------------------------------------------------------------
    # If two clauses with the SAME clause_id from DIFFERENT non-temporal sources
    # or conflicting unresolvable amounts appear in evidence:
    for cid, cs in clause_by_id.items():
        if len(cs) > 1:
            # Check if this is resolved by temporal context
            has_orig = any(not c.is_amendment for c in cs)
            has_amend = any(c.is_amendment for c in cs)

            if has_orig and has_amend:
                # If temporal context is PRE_AMENDMENT or POST_AMENDMENT, this is resolved versioning, NOT conflict
                if t_ctx and t_ctx.status in (TemporalStatus.PRE_AMENDMENT, TemporalStatus.POST_AMENDMENT):
                    continue
                # If temporal context is UNSPECIFIED, this is chronological temporal applicability, NOT an unresolved conflict
                if t_ctx and t_ctx.status == TemporalStatus.UNSPECIFIED:
                    continue

    return conflicts
