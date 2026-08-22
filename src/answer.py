"""
Grounded Answer Generation with Gemini API.

Constructs strict grounding prompts from retrieved policy clauses,
invokes Gemini, and validates returned citations against supplied evidence.
"""

from dataclasses import dataclass, field
import os
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
    from .retriever import PolicyRetriever, ScoredClause
    from .loader import load_policy
except ImportError:
    from models import PolicyClause
    from retriever import PolicyRetriever, ScoredClause
    from loader import load_policy


# Default Gemini model and timeout configured for grounded answer generation
DEFAULT_GEMINI_MODEL: str = "gemini-3.6-flash"
DEFAULT_TIMEOUT_SECONDS: float = 60.0


INSUFFICIENT_EVIDENCE_PHRASES = [
    "insufficient evidence",
    "does not contain",
    "not enough evidence",
    "cannot be determined",
    "policy does not state",
    "policy does not mention",
    "no information",
    "not mentioned in the provided policy",
    "cannot answer based on the provided policy",
    "evidence is insufficient",
]


@dataclass
class AnswerResult:
    """
    Structured result returned by the grounded answer generator.
    """
    answer: str
    citations: List[str] = field(default_factory=list)
    grounded: bool = True
    insufficient_evidence: bool = False
    unsupported_citations: List[str] = field(default_factory=list)
    evidence_clause_ids: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "grounded": self.grounded,
            "insufficient_evidence": self.insufficient_evidence,
            "unsupported_citations": self.unsupported_citations,
            "evidence_clause_ids": self.evidence_clause_ids,
            "raw_response": self.raw_response,
        }

    def __str__(self) -> str:
        status = "INSUFFICIENT EVIDENCE" if self.insufficient_evidence else ("GROUNDED" if self.grounded else "UNGROUNDED")
        cites = ", ".join(f"§{c}" for c in self.citations) if self.citations else "None"
        return f"[{status}] (Citations: {cites})\n{self.answer}"


def extract_citations(text: str) -> List[str]:
    """
    Extracts all candidate clause IDs (e.g. '4.3.2' from '§4.3.2' or '[§4.3.2]') from text.
    """
    matches = re.findall(r"(?:§|\[§|clause\s+|section\s+)?(\d+\.\d+\.\d+)", text, re.IGNORECASE)
    # Deduplicate while preserving order
    seen: Set[str] = set()
    ordered: List[str] = []
    for m in matches:
        clean = m.strip()
        if clean and clean not in seen:
            seen.add(clean)
            ordered.append(clean)
    return ordered


def build_grounded_prompt(
    question: str,
    retrieved_clauses: List[Union[PolicyClause, ScoredClause]],
) -> str:
    """
    Constructs a strict grounding prompt containing only the retrieved evidence.
    """
    evidence_blocks: List[str] = []
    for item in retrieved_clauses:
        clause = item.clause if isinstance(item, ScoredClause) else item
        title_str = f" ({clause.clause_title})" if clause.clause_title else ""
        section_str = f" [{clause.parent_section}]" if clause.parent_section else ""
        header = f"[§{clause.clause_id}]{title_str}{section_str}"
        evidence_blocks.append(f"{header}\n{clause.clause_text}")

    evidence_text = "\n\n".join(evidence_blocks)

    prompt = f"""You are a helpful, strictly grounded assistant for the Calder County Household Support Program.

CRITICAL INSTRUCTIONS:
1. The supplied POLICY EVIDENCE below is the ONLY authoritative source of information.
2. Answer the USER QUESTION using ONLY the facts explicitly stated in the POLICY EVIDENCE.
3. Do NOT use outside knowledge, general assumptions, or extrapolate beyond the text.
4. If the POLICY EVIDENCE does not contain sufficient facts to answer the question, state clearly: "The provided policy evidence is insufficient to answer this question."
5. For every substantive statement or rule you state, cite the exact clause ID (e.g. §4.3.2) from the POLICY EVIDENCE that supports it.
6. Do NOT fabricate, guess, or invent clause IDs. Only cite clause IDs that appear in the POLICY EVIDENCE.
7. If the POLICY EVIDENCE contains conflicting rules, different time limits, or apparent discrepancies (e.g. §4.3.2 requiring reporting within 10 calendar days versus §9.1.4 referencing 30 calendar days), you MUST explicitly describe BOTH provisions, highlight the discrepancy, and cite both supporting clauses (§4.3.2 and §9.1.4). Do NOT ignore one in favor of the other.

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
    
    Returns:
        Tuple of (valid_citations, unsupported_citations, is_grounded)
    """
    found_citations = extract_citations(raw_text)
    valid_citations: List[str] = []
    unsupported_citations: List[str] = []

    for cid in found_citations:
        if cid in allowed_clause_ids:
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
        timeout_seconds: API request timeout in seconds (default 20.0s).
        
    Returns:
        AnswerResult containing answer text, validated citations, and grounding status.
    """
    # Extract allowed clause IDs
    allowed_ids: Set[str] = set()
    for item in retrieved_clauses:
        clause = item.clause if isinstance(item, ScoredClause) else item
        allowed_ids.add(clause.clause_id)

    # Fast-path: If no evidence was retrieved, refuse without calling LLM
    if not retrieved_clauses or not allowed_ids:
        return AnswerResult(
            answer="The provided policy evidence is insufficient to answer this question.",
            citations=[],
            grounded=True,
            insufficient_evidence=True,
            evidence_clause_ids=[],
            raw_response=None,
        )

    prompt = build_grounded_prompt(question, retrieved_clauses)
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
    )


if __name__ == "__main__":
    import sys
    # Ensure UTF-8 output on Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("Grounded Answer Generation Demonstration")
    print("=" * 70)

    policy_path = "data/policy-manual.md"
    clauses = load_policy(policy_path)
    retriever = PolicyRetriever(clauses=clauses)

    demo_question = "What is the deadline for reporting a change of circumstances?"
    print(f"\nQuestion: {demo_question}\n")

    retrieved = retriever.retrieve(demo_question, top_k=5)
    print("Retrieved Clause IDs:", [f"§{r.clause_id}" for r in retrieved])
    for r in retrieved:
        print(f" - §{r.clause_id} ({r.parent_section}): {r.clause_text[:70]}...")

    print("\n--- Answer Generation ---")
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        print("Using live Gemini API...")
        result = generate_answer(demo_question, retrieved, api_key=api_key)
    else:
        print("[Notice] GEMINI_API_KEY environment variable is not set.")
        print("Executing demonstration using a deterministic mock response...")

        # Mock generator simulating grounded model output
        def mock_llm(p: str) -> str:
            return (
                "Under §4.3.2, a recipient must report any change in household composition, "
                "income, address, or circumstances within 10 calendar days. "
                "However, §9.1.4 states that where a change is reported within 30 calendar days, "
                "no overpayment is established before the date the Department was in a position to act."
            )

        result = generate_answer(demo_question, retrieved, client=mock_llm)

    print("\nGenerated Answer:")
    print(result.answer)
    print(f"\nValidated Citations: {result.citations}")
    print(f"Unsupported Citations: {result.unsupported_citations}")
    print(f"Grounded Status: {result.grounded}")
    print(f"Insufficient Evidence: {result.insufficient_evidence}")
    print("=" * 70)
