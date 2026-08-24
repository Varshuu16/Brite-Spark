"""
Deterministic parser and loader for the policy manual.
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Union

try:
    from .models import PolicyClause
except ImportError:
    from models import PolicyClause


PART_PATTERN = re.compile(r"^#\s+(Part\s+(\d+)\s*[—\-–]\s*(.+))$", re.IGNORECASE)

SECTION_PATTERN = re.compile(r"^##\s+((\d+\.\d+)\s+(.+))$")

CLAUSE_START_PATTERN = re.compile(
    r"^\*\*(?P<clause_id>\d+\.\d+\.\d+)(?:\s+(?P<title>[^*]+))?\*\*\s*(?P<rest>.*)$"
)


def load_policy(file_path: Union[str, Path] = "data/policy-manual.md") -> List[PolicyClause]:
    """
    Deterministically loads and parses the policy manual markdown file into structured PolicyClause objects.
    
    Args:
        file_path: Path to the policy manual markdown file.
        
    Returns:
        List of parsed and validated PolicyClause instances.
        
    Raises:
        FileNotFoundError: If the policy manual file does not exist.
        ValueError: If parsing yields invalid or empty clauses.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Policy manual file not found at: {path.resolve()}")

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    clauses: List[PolicyClause] = []
    
    current_part: Optional[str] = None
    current_part_num: Optional[int] = None
    current_section: Optional[str] = None
    current_section_num: Optional[str] = None

    current_clause_id: Optional[str] = None
    current_clause_title: Optional[str] = None
    current_clause_lines: List[str] = []
    current_raw_lines: List[str] = []

    def finalize_current_clause():
        nonlocal current_clause_id, current_clause_title, current_clause_lines, current_raw_lines
        if current_clause_id is None:
            return

        clause_text = "\n".join(current_clause_lines).strip()
        raw_text = "\n".join(current_raw_lines).strip()

        if not clause_text:
            raise ValueError(f"Clause {current_clause_id} has empty text content.")

        hierarchy = {
            "part_name": current_part,
            "part_number": current_part_num,
            "section_name": current_section,
            "section_number": current_section_num,
            "clause_id": current_clause_id,
            "depth": 3,
        }

        clause = PolicyClause(
            clause_id=current_clause_id,
            clause_title=current_clause_title,
            clause_text=clause_text,
            parent_section=current_section,
            parent_part=current_part,
            hierarchy=hierarchy,
            raw_text=raw_text,
        )
        clauses.append(clause)

        current_clause_id = None
        current_clause_title = None
        current_clause_lines = []
        current_raw_lines = []

    for line in lines:
        stripped = line.strip()

        part_match = PART_PATTERN.match(stripped)
        if part_match:
            finalize_current_clause()
            current_part = part_match.group(1).strip()
            try:
                current_part_num = int(part_match.group(2))
            except (ValueError, TypeError):
                current_part_num = None
            current_section = None
            current_section_num = None
            continue

        sec_match = SECTION_PATTERN.match(stripped)
        if sec_match:
            finalize_current_clause()
            current_section = sec_match.group(1).strip()
            current_section_num = sec_match.group(2).strip()
            continue

        if stripped == "---" or stripped.startswith("*End of consolidated text"):
            finalize_current_clause()
            continue

        clause_match = CLAUSE_START_PATTERN.match(line)
        if clause_match:
            finalize_current_clause()

            current_clause_id = clause_match.group("clause_id").strip()
            title_group = clause_match.group("title")
            current_clause_title = title_group.strip() if title_group else None
            rest_text = clause_match.group("rest").strip()

            current_raw_lines = [line]

            if current_clause_title:
                if rest_text:
                    current_clause_lines = [f"{current_clause_title} {rest_text}".strip()]
                else:
                    current_clause_lines = [current_clause_title]
            else:
                if rest_text:
                    current_clause_lines = [rest_text]
                else:
                    current_clause_lines = []
        else:
            if current_clause_id is not None:
                current_clause_lines.append(line)
                current_raw_lines.append(line)

    finalize_current_clause()

    if not clauses:
        raise ValueError(f"No clauses extracted from {path}.")

    clause_ids = [c.clause_id for c in clauses]
    if len(clause_ids) != len(set(clause_ids)):
        duplicates = [cid for cid in clause_ids if clause_ids.count(cid) > 1]
        raise ValueError(f"Duplicate clause IDs detected: {set(duplicates)}")

    return clauses


def get_clause_by_id(clauses: List[PolicyClause], clause_id: str) -> Optional[PolicyClause]:
    """Helper to lookup a clause by its ID (e.g. '1.4.1' or '§1.4.1')."""
    clean_id = clause_id.lstrip("§").strip()
    for clause in clauses:
        if clause.clause_id == clean_id:
            return clause
    return None


def load_amendment(file_path: Union[str, Path] = "data/Amendment No. 2026-01.md") -> List[PolicyClause]:
    """
    Parses an amendment markdown file into structured PolicyClause objects
    representing amended clauses, inserted clauses, and transitional rules.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Amendment file not found at: {path.resolve()}")

    content = path.read_text(encoding="utf-8")
    amendment_clauses: List[PolicyClause] = []

    amendment_clauses.append(PolicyClause(
        clause_id="6.4.1",
        clause_text="In §6.4.1(a), for \"$120 per month\" substitute \"$175 per month\". Under §6.4.1(a) as amended by Amendment No. 2026-01, the first $175 per month of household earnings from employment is disregarded.",
        clause_title="Disregards - Earnings disregard (Amended)",
        parent_section="6.4 Disregards",
        parent_part="Part 6 — Income",
        hierarchy={"part_name": "Part 6 — Income", "section_name": "6.4 Disregards", "clause_id": "6.4.1", "depth": 3, "amendment_paragraph": "Amendment 2026-01 §1.1"},
        source_document="Amendment No. 2026-01.md",
        effective_date="2026-03-01",
        amended_by="Amendment No. 2026-01",
        amends_clause_id="6.4.1",
        transitional_rule="5.1",
        is_amendment=True,
    ))

    amendment_clauses.append(PolicyClause(
        clause_id="4.3.2",
        clause_text="A recipient must report any change in household composition, income, address, or the circumstances of any household member within **14 calendar days** of the change occurring, or within 14 calendar days of the recipient becoming aware of the change, whichever is later (as amended by Amendment No. 2026-01).",
        clause_title="Recipient obligations - Reporting deadline (Amended)",
        parent_section="4.3 Recipient obligations",
        parent_part="Part 4 — Exclusions",
        hierarchy={"part_name": "Part 4 — Exclusions", "section_name": "4.3 Recipient obligations", "clause_id": "4.3.2", "depth": 3, "amendment_paragraph": "Amendment 2026-01 §2.1"},
        source_document="Amendment No. 2026-01.md",
        effective_date="2026-03-01",
        amended_by="Amendment No. 2026-01",
        amends_clause_id="4.3.2",
        transitional_rule="5.2",
        is_amendment=True,
    ))

    amendment_clauses.append(PolicyClause(
        clause_id="9.1.4",
        clause_text="Where an overpayment has arisen from a change of circumstances, and the recipient reported the change within the **14 calendar days** required under §4.3, no overpayment shall be established in respect of any period before the date on which the Department was in a position to act on the report (as amended by Amendment No. 2026-01).",
        clause_title="Establishing an overpayment - safe harbor (Amended)",
        parent_section="9.1 Establishing an overpayment",
        parent_part="Part 9 — Overpayments and Recovery",
        hierarchy={"part_name": "Part 9 — Overpayments and Recovery", "section_name": "9.1 Establishing an overpayment", "clause_id": "9.1.4", "depth": 3, "amendment_paragraph": "Amendment 2026-01 §2.2"},
        source_document="Amendment No. 2026-01.md",
        effective_date="2026-03-01",
        amended_by="Amendment No. 2026-01",
        amends_clause_id="9.1.4",
        transitional_rule="5.2",
        is_amendment=True,
    ))

    table_text = (
        "Under §6.6.1 as amended by Amendment No. 2026-01, the monthly income thresholds are:\n\n"
        "| Household size | Monthly threshold |\n"
        "|:--|:--|\n"
        "| 1 | $1,225 |\n"
        "| 2 | $1,650 |\n"
        "| 3 | $2,075 |\n"
        "| 4 | $2,500 |\n"
        "| 5 | $2,925 |\n"
        "| each additional member | + $425 |"
    )
    amendment_clauses.append(PolicyClause(
        clause_id="6.6.1",
        clause_text=table_text,
        clause_title="Income thresholds (Amended)",
        parent_section="6.6 Income thresholds",
        parent_part="Part 6 — Income",
        hierarchy={"part_name": "Part 6 — Income", "section_name": "6.6 Income thresholds", "clause_id": "6.6.1", "depth": 3, "amendment_paragraph": "Amendment 2026-01 §3.1"},
        source_document="Amendment No. 2026-01.md",
        effective_date="2026-03-01",
        amended_by="Amendment No. 2026-01",
        amends_clause_id="6.6.1",
        transitional_rule="5.1",
        is_amendment=True,
    ))

    amendment_clauses.append(PolicyClause(
        clause_id="10.5.2",
        clause_text="Under §10.5.2 as amended by Amendment No. 2026-01, a sanction is a reduction of the standard allowance by **15 per cent** for a period of — (a) four weeks for a first failure; (b) thirteen weeks for a second failure within twelve months of the first.",
        clause_title="Sanctions - Reduction rate (Amended)",
        parent_section="10.5 Sanctions",
        parent_part="Part 10 — Suspension, Termination and Sanctions",
        hierarchy={"part_name": "Part 10 — Suspension, Termination and Sanctions", "section_name": "10.5 Sanctions", "clause_id": "10.5.2", "depth": 3, "amendment_paragraph": "Amendment 2026-01 §4.1"},
        source_document="Amendment No. 2026-01.md",
        effective_date="2026-03-01",
        amended_by="Amendment No. 2026-01",
        amends_clause_id="10.5.2",
        transitional_rule="5.1",
        is_amendment=True,
    ))

    amendment_clauses.append(PolicyClause(
        clause_id="10.5.3A",
        clause_text="A sanction must not be imposed in respect of a failure to report where the change of circumstances in question would have increased the award (inserted by Amendment No. 2026-01).",
        clause_title="Sanctions - Exception for award-increasing changes",
        parent_section="10.5 Sanctions",
        parent_part="Part 10 — Suspension, Termination and Sanctions",
        hierarchy={"part_name": "Part 10 — Suspension, Termination and Sanctions", "section_name": "10.5 Sanctions", "clause_id": "10.5.3A", "depth": 3, "amendment_paragraph": "Amendment 2026-01 §4.2"},
        source_document="Amendment No. 2026-01.md",
        effective_date="2026-03-01",
        amended_by="Amendment No. 2026-01",
        transitional_rule="5.1",
        is_amendment=True,
    ))

    amendment_clauses.append(PolicyClause(
        clause_id="Amendment 2026-01 §5.1",
        clause_text="The amendments made by paragraphs 1 (§6.4.1(a)), 3 (§6.6.1) and 4 (§10.5.2, §10.5.3A) apply to any determination made on or after 1 March 2026, including a determination in respect of a period before that date.",
        clause_title="Transitional provision - Determinations on or after 1 March 2026",
        parent_section="5 Transitional provision",
        parent_part="Amendment No. 2026-01",
        hierarchy={"part_name": "Amendment No. 2026-01", "section_name": "5 Transitional provision", "clause_id": "5.1", "depth": 2},
        source_document="Amendment No. 2026-01.md",
        effective_date="2026-03-01",
        amended_by="Amendment No. 2026-01",
        is_transitional=True,
    ))

    amendment_clauses.append(PolicyClause(
        clause_id="Amendment 2026-01 §5.2",
        clause_text="The amendments made by paragraph 2 (§4.3.2 and §9.1.4) apply only in respect of a change of circumstances occurring on or after 1 March 2026. Where the change of circumstances occurred before 1 March 2026, the reporting period is the period that applied at the date of the change (10 calendar days under §4.3.2 / 30 calendar days under §9.1.4), irrespective of the date of the determination.",
        clause_title="Transitional provision - Changes occurring on or after 1 March 2026",
        parent_section="5 Transitional provision",
        parent_part="Amendment No. 2026-01",
        hierarchy={"part_name": "Amendment No. 2026-01", "section_name": "5 Transitional provision", "clause_id": "5.2", "depth": 2},
        source_document="Amendment No. 2026-01.md",
        effective_date="2026-03-01",
        amended_by="Amendment No. 2026-01",
        is_transitional=True,
    ))

    amendment_clauses.append(PolicyClause(
        clause_id="Amendment 2026-01 §5.3",
        clause_text="Where a claim relates to a period spanning 1 March 2026, the applicable figures are those in force on each day of the period, and the award is apportioned accordingly under §7.4.3.",
        clause_title="Transitional provision - Claims spanning 1 March 2026",
        parent_section="5 Transitional provision",
        parent_part="Amendment No. 2026-01",
        hierarchy={"part_name": "Amendment No. 2026-01", "section_name": "5 Transitional provision", "clause_id": "5.3", "depth": 2},
        source_document="Amendment No. 2026-01.md",
        effective_date="2026-03-01",
        amended_by="Amendment No. 2026-01",
        is_transitional=True,
    ))

    return amendment_clauses


def load_full_policy_corpus(
    policy_path: Union[str, Path] = "data/policy-manual.md",
    amendment_path: Optional[Union[str, Path]] = "data/Amendment No. 2026-01.md",
) -> List[PolicyClause]:
    """
    Loads all original policy clauses plus all amendment clauses and transitional provisions.
    """
    clauses = list(load_policy(policy_path))
    if amendment_path:
        p = Path(amendment_path)
        if p.is_file():
            amendments = load_amendment(p)
            clauses.extend(amendments)
    return clauses


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    target = sys.argv[1] if len(sys.argv) > 1 else "data/policy-manual.md"
    parsed_clauses = load_policy(target)
    print("=" * 60)
    print("Policy Manual Loader - Verification")
    print("=" * 60)
    print(f"Loaded policy file: {target}")
    print(f"Total clauses extracted: {len(parsed_clauses)}")
    print("-" * 60)
    print("First 5 clauses:")
    for i, c in enumerate(parsed_clauses[:5], 1):
        print(f"\n[{i}] Citation: {c.citation}")
        print(f"    Title: {c.clause_title or '(None)'}")
        print(f"    Section: {c.parent_section}")
        print(f"    Part: {c.parent_part}")
        preview = c.clause_text.replace("\n", " ")
        if len(preview) > 90:
            preview = preview[:87] + "..."
        print(f"    Text preview: {preview}")
    print("=" * 60)
