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


# Regex to detect `# Part <num> — <title>`
PART_PATTERN = re.compile(r"^#\s+(Part\s+(\d+)\s*[—\-–]\s*(.+))$", re.IGNORECASE)

# Regex to detect `## <num>.<num> <title>`
SECTION_PATTERN = re.compile(r"^##\s+((\d+\.\d+)\s+(.+))$")

# Regex to detect clause start: `**1.1.1** ...` or `**1.4.1 Applicant** — ...`
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

        # Reset state
        current_clause_id = None
        current_clause_title = None
        current_clause_lines = []
        current_raw_lines = []

    for line in lines:
        stripped = line.strip()

        # Check for Part heading (# Part ...)
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

        # Check for Section heading (## X.Y ...)
        sec_match = SECTION_PATTERN.match(stripped)
        if sec_match:
            finalize_current_clause()
            current_section = sec_match.group(1).strip()
            current_section_num = sec_match.group(2).strip()
            continue

        # Check for divider or footer
        if stripped == "---" or stripped.startswith("*End of consolidated text"):
            finalize_current_clause()
            continue

        # Check for Clause start (**X.Y.Z** or **X.Y.Z Title**)
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
            # Continuing lines within the current clause
            if current_clause_id is not None:
                current_clause_lines.append(line)
                current_raw_lines.append(line)

    # Finalize last clause if any
    finalize_current_clause()

    # Validate extracted clauses
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


if __name__ == "__main__":
    import sys
    # Ensure UTF-8 output in Windows console if supported
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
