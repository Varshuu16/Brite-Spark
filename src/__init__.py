"""
BriteSpark Problem 1 — Deterministic Policy Engine & Grounded Answer System
"""

from .models import PolicyClause, AmendmentProvision, PolicyCitation, PolicyConflict
from .loader import load_policy, load_amendment, load_full_policy_corpus, get_clause_by_id
from .retriever import PolicyRetriever, ScoredClause
from .temporal import (
    TemporalContext,
    TemporalStatus,
    QueryEventType,
    extract_temporal_context,
)
from .conflict import detect_conflicts
from .answer import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    AnswerResult,
    generate_answer,
    build_grounded_prompt,
    validate_citations,
    validate_citations_detailed,
    extract_citations,
    check_citation_completeness,
)

__all__ = [
    "PolicyClause",
    "AmendmentProvision",
    "PolicyCitation",
    "PolicyConflict",
    "load_policy",
    "load_amendment",
    "load_full_policy_corpus",
    "get_clause_by_id",
    "PolicyRetriever",
    "ScoredClause",
    "TemporalContext",
    "TemporalStatus",
    "QueryEventType",
    "extract_temporal_context",
    "detect_conflicts",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "AnswerResult",
    "generate_answer",
    "build_grounded_prompt",
    "validate_citations",
    "validate_citations_detailed",
    "extract_citations",
    "check_citation_completeness",
]
