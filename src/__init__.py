"""
BriteSpark Problem 1 — Deterministic Policy Engine & Grounded Answer System
"""

from .models import PolicyClause, AmendmentProvision
from .loader import load_policy, load_amendment, load_full_policy_corpus, get_clause_by_id
from .retriever import PolicyRetriever, ScoredClause
from .temporal import (
    TemporalContext,
    TemporalStatus,
    QueryEventType,
    extract_temporal_context,
)
from .answer import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    AnswerResult,
    generate_answer,
    build_grounded_prompt,
    validate_citations,
    extract_citations,
)

__all__ = [
    "PolicyClause",
    "AmendmentProvision",
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
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "AnswerResult",
    "generate_answer",
    "build_grounded_prompt",
    "validate_citations",
    "extract_citations",
]
