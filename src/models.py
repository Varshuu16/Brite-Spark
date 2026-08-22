"""
Data models for policy document representations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PolicyClause:
    """
    Represents a discrete, citeable policy clause extracted from the policy manual.
    """
    clause_id: str
    clause_text: str
    clause_title: Optional[str] = None
    parent_section: Optional[str] = None
    parent_part: Optional[str] = None
    hierarchy: Dict[str, Any] = field(default_factory=dict)
    raw_text: Optional[str] = None

    @property
    def citation(self) -> str:
        """Returns the canonical section citation string, e.g. '§1.4.1'."""
        return f"§{self.clause_id}"

    @property
    def full_reference(self) -> str:
        """Returns a human-readable full reference including part and section context."""
        parts = []
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
        }

    def __str__(self) -> str:
        title_str = f" ({self.clause_title})" if self.clause_title else ""
        return f"[{self.citation}{title_str}] {self.clause_text}"
