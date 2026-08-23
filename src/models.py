"""
Data models for policy document representations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PolicyClause:
    """
    Represents a discrete, citeable policy clause extracted from the policy manual
    or an amendment document, with temporal validity metadata.
    """
    clause_id: str
    clause_text: str
    clause_title: Optional[str] = None
    parent_section: Optional[str] = None
    parent_part: Optional[str] = None
    hierarchy: Dict[str, Any] = field(default_factory=dict)
    raw_text: Optional[str] = None
    # Temporal & Amendment Metadata
    source_document: str = "policy-manual.md"
    effective_date: Optional[str] = "2025-12-31"
    amended_by: Optional[str] = None
    amends_clause_id: Optional[str] = None
    transitional_rule: Optional[str] = None
    is_amendment: bool = False
    is_transitional: bool = False

    @property
    def citation(self) -> str:
        """Returns the canonical section citation string, e.g. '§1.4.1' or 'Amendment 2026-01 §5.2'."""
        if self.is_transitional:
            if self.clause_id.startswith("Amendment"):
                return self.clause_id
            if self.amended_by:
                return f"{self.amended_by} §{self.clause_id}"
        return f"§{self.clause_id}"

    @property
    def full_reference(self) -> str:
        """Returns a human-readable full reference including part, section context, and amendment info."""
        parts = []
        if self.amended_by:
            parts.append(f"[{self.amended_by}]")
        if self.parent_part:
            parts.append(self.parent_part)
        if self.parent_section:
            parts.append(self.parent_section)
        parts.append(self.citation + (f" ({self.clause_title})" if self.clause_title else ""))
        return " > ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the PolicyClause into a serializable dictionary."""
        return {
            "clause_id": self.clause_id,
            "clause_title": self.clause_title,
            "clause_text": self.clause_text,
            "parent_section": self.parent_section,
            "parent_part": self.parent_part,
            "hierarchy": self.hierarchy,
            "citation": self.citation,
            "full_reference": self.full_reference,
            "raw_text": self.raw_text,
            "source_document": self.source_document,
            "effective_date": self.effective_date,
            "amended_by": self.amended_by,
            "amends_clause_id": self.amends_clause_id,
            "transitional_rule": self.transitional_rule,
            "is_amendment": self.is_amendment,
            "is_transitional": self.is_transitional,
        }

    def __str__(self) -> str:
        title_str = f" ({self.clause_title})" if self.clause_title else ""
        amend_str = f" [{self.amended_by}]" if self.amended_by else ""
        return f"[{self.citation}{title_str}{amend_str}] {self.clause_text}"


@dataclass
class AmendmentProvision:
    """
    Represents a specific provision within an amendment document.
    """
    amendment_id: str
    paragraph_id: str
    title: str
    target_clause_id: Optional[str]
    action: str  # 'SUBSTITUTE', 'INSERT', 'TRANSITIONAL'
    text: str
    effective_date: str = "2026-03-01"
    transitional_rule: Optional[str] = None
