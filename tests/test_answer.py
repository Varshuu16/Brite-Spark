"""
Unit tests for grounded answer generation, citation validation, and prompt construction.
"""

from pathlib import Path
import unittest

from src.models import PolicyClause
from src.loader import load_policy, get_clause_by_id
from src.retriever import ScoredClause
from src.answer import (
    DEFAULT_GEMINI_MODEL,
    AnswerResult,
    generate_answer,
    build_grounded_prompt,
    validate_citations,
    extract_citations,
    check_insufficient_evidence,
)


class TestGroundedAnswer(unittest.TestCase):
    """Test suite for answer generation logic, grounding guarantees, and citation verification."""

    @classmethod
    def setUpClass(cls):
        cls.policy_path = Path("data/policy-manual.md")
        cls.clauses = load_policy(cls.policy_path)
        cls.clause_6_4_1 = get_clause_by_id(cls.clauses, "6.4.1")
        cls.clause_4_3_2 = get_clause_by_id(cls.clauses, "4.3.2")
        cls.clause_1_4_6 = get_clause_by_id(cls.clauses, "1.4.6")

    def test_grounded_answer(self):
        """Test A: Grounded answer with valid citations is marked grounded."""
        evidence = [self.clause_6_4_1]
        mock_response = "Under §6.4.1, the first $120 per month of household earnings is disregarded."

        result = generate_answer(
            question="How much is the earnings disregard?",
            retrieved_clauses=evidence,
            client=lambda prompt: mock_response,
        )

        self.assertTrue(result.grounded)
        self.assertFalse(result.insufficient_evidence)
        self.assertEqual(result.citations, ["6.4.1"])
        self.assertEqual(result.unsupported_citations, [])
        self.assertIn("120", result.answer)

    def test_citation_validation_accepted(self):
        """Test B: Citing an available retrieved clause is accepted."""
        evidence = [
            ScoredClause(clause=self.clause_4_3_2, score=10.0),
            ScoredClause(clause=self.clause_1_4_6, score=5.0),
        ]
        mock_response = "According to §4.3.2 and §1.4.6, reports must be made within 10 days."

        result = generate_answer(
            question="What is the deadline for reporting?",
            retrieved_clauses=evidence,
            client=lambda prompt: mock_response,
        )

        self.assertTrue(result.grounded)
        self.assertIn("4.3.2", result.citations)
        self.assertIn("1.4.6", result.citations)
        self.assertEqual(result.unsupported_citations, [])

    def test_unsupported_citation_rejected(self):
        """Test C: Citing a clause NOT supplied in the evidence is flagged as unsupported/ungrounded."""
        evidence = [self.clause_6_4_1]  # Only 6.4.1 provided
        # Model hallucinates a citation to §9.9.9 and §4.3.2 which were not in evidence
        mock_response = "Under §6.4.1 earnings are disregarded, but §9.9.9 and §4.3.2 apply for appeals."

        result = generate_answer(
            question="What are the earnings rules?",
            retrieved_clauses=evidence,
            client=lambda prompt: mock_response,
        )

        self.assertFalse(result.grounded, "Answer with hallucinated citations must not be marked grounded")
        self.assertEqual(result.citations, ["6.4.1"])
        self.assertIn("9.9.9", result.unsupported_citations)
        self.assertIn("4.3.2", result.unsupported_citations)

    def test_insufficient_evidence(self):
        """Test D: When evidence does not answer question, result indicates insufficient evidence."""
        evidence = [self.clause_1_4_6]  # Evidence only about full-time students
        mock_response = "The provided policy evidence is insufficient to answer how to appeal a property tax assessment."

        result = generate_answer(
            question="How do I appeal my property tax?",
            retrieved_clauses=evidence,
            client=lambda prompt: mock_response,
        )

        self.assertTrue(result.insufficient_evidence)
        self.assertTrue(result.grounded)

    def test_no_evidence_fast_refusal(self):
        """Test E: Empty retrieval results return immediate refusal without invoking LLM."""
        called = False

        def failing_mock(prompt):
            nonlocal called
            called = True
            return "Hallucinated answer"

        result = generate_answer(
            question="What is the speed of light in Calder County?",
            retrieved_clauses=[],
            client=failing_mock,
        )

        self.assertFalse(called, "LLM must not be called when evidence is empty")
        self.assertTrue(result.insufficient_evidence)
        self.assertTrue(result.grounded)
        self.assertEqual(len(result.citations), 0)
        self.assertIn("insufficient", result.answer.lower())

    def test_prompt_grounding_structure(self):
        """Test F: Grounded prompt contains question, instructions, and exact retrieved text."""
        question = "What constitutes a full-time student?"
        evidence = [self.clause_1_4_6]

        prompt = build_grounded_prompt(question, evidence)

        self.assertIn("USER QUESTION:", prompt)
        self.assertIn(question, prompt)
        self.assertIn("POLICY EVIDENCE:", prompt)
        self.assertIn("[§1.4.6]", prompt)
        self.assertIn("Full-time student", prompt)
        self.assertIn("enrolled in a course of study at an accredited institution", prompt)
        self.assertIn("The supplied POLICY EVIDENCE below is the ONLY authoritative source", prompt)

    def test_citation_extraction_utility(self):
        """Test citation extraction utility regex handles multiple formats."""
        sample_text = "See §1.4.1, [§4.3.2], clause 6.4.1, and Section 10.5.2 for details."
        extracted = extract_citations(sample_text)
        self.assertEqual(extracted, ["1.4.1", "4.3.2", "6.4.1", "10.5.2"])

    def test_default_model_configuration(self):
        """Test that default Gemini model is configured to gemini-3.6-flash and passed to model invocation."""
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-3.6-flash")

        # Mock client to verify model parameter received
        class MockModels:
            def __init__(self):
                self.received_model = None

            def generate_content(self, model, contents, config=None):
                self.received_model = model
                class Response:
                    text = "Mock response under §6.4.1."
                return Response()

        class MockClient:
            def __init__(self):
                self.models = MockModels()

        mock_client = MockClient()
        result = generate_answer(
            question="What is the disregard?",
            retrieved_clauses=[self.clause_6_4_1],
            client=mock_client,
        )
        self.assertEqual(mock_client.models.received_model, "gemini-3.6-flash")

    def test_api_error_graceful_handling(self):
        """Test that API exceptions/timeouts are caught gracefully and return structured failure."""
        class FailingModels:
            def generate_content(self, model, contents, config=None):
                raise TimeoutError("Request timed out after 20.0 seconds")

        class FailingClient:
            def __init__(self):
                self.models = FailingModels()

        result = generate_answer(
            question="What is the rule under §6.4.1?",
            retrieved_clauses=[self.clause_6_4_1],
            client=FailingClient(),
        )

        self.assertFalse(result.grounded)
        self.assertTrue(result.insufficient_evidence)
        self.assertIn("Error communicating with Gemini API", result.answer)
        self.assertIn("timed out", result.answer)
        self.assertEqual(result.citations, [])


if __name__ == "__main__":
    unittest.main()
