"""
BriteSpark 2026 - Problem 1: The Grounded Answer
Calder County Household Support Program Policy Assistant
"""

from .models import PolicyClause
from .loader import load_policy
from .retriever import PolicyRetriever, ScoredClause
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
    "load_policy",
    "PolicyRetriever",
    "ScoredClause",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "AnswerResult",
    "generate_answer",
    "build_grounded_prompt",
    "validate_citations",
    "extract_citations",
]
