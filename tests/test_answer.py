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
        evidence = [self.clause_6_4_1]
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
        evidence = [self.clause_1_4_6]  
        mock_response = "The provided policy evidence is insufficient to answer how to appeal a property tax assessment."

        result = generate_answer(
            question="How do I appeal my property tax?",
            retrieved_clauses=evidence,
            client=lambda prompt: mock_response,
        )

        self.assertTrue(result.insufficient_evidence)
        self.assertFalse(result.grounded)

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
        self.assertFalse(result.grounded)
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

    def test_temporal_answer_february_2026_reporting(self):
        """Test C: February 2026 change generates grounded answer with 10-day deadline."""
        from src.loader import load_full_policy_corpus
        from src.retriever import PolicyRetriever

        corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        retriever = PolicyRetriever(clauses=corpus)

        question = "What is the deadline for reporting a change of circumstances that occurred on 10 February 2026?"
        retrieved = retriever.retrieve(question, top_k=4)

        def mock_llm(prompt: str) -> str:
            return (
                "For a change of circumstances occurring on 10 February 2026, under Amendment No. 2026-01 §5.2, "
                "the pre-amendment reporting deadline applies. Under §4.3.2, the recipient must report the change "
                "within **10 calendar days** of the change occurring."
            )

        result = generate_answer(question, retrieved, client=mock_llm)
        self.assertTrue(result.grounded)
        self.assertIn("4.3.2", result.citations)
        self.assertIn("10 calendar days", result.answer)

    def test_temporal_answer_april_2026_reporting(self):
        """Test C: April 2026 change generates grounded answer with amended 14-day deadline."""
        from src.loader import load_full_policy_corpus
        from src.retriever import PolicyRetriever

        corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        retriever = PolicyRetriever(clauses=corpus)

        question = "What is the deadline for reporting a change of circumstances that occurred on 15 April 2026?"
        retrieved = retriever.retrieve(question, top_k=4)

        def mock_llm(prompt: str) -> str:
            return (
                "For a change of circumstances occurring on 15 April 2026, under §4.3.2 as amended by "
                "Amendment No. 2026-01 §5.2, the recipient must report within **14 calendar days**."
            )

        result = generate_answer(question, retrieved, client=mock_llm)
        self.assertTrue(result.grounded)
        self.assertIn("4.3.2", result.citations)
        self.assertIn("14 calendar days", result.answer)

    def test_temporal_answer_transitional_determination_5_1(self):
        """Test F: Determination made on/after 1 March 2026 applies $175 disregard under §5.1."""
        from src.loader import load_full_policy_corpus
        from src.retriever import PolicyRetriever

        corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        retriever = PolicyRetriever(clauses=corpus)

        question = "What is the earnings disregard for a determination made on 15 March 2026 for a January 2026 claim?"
        retrieved = retriever.retrieve(question, top_k=4)

        def mock_llm(prompt: str) -> str:
            return (
                "Under Amendment No. 2026-01 §5.1, the amendments apply to any determination made on or after "
                "1 March 2026, even for prior periods. Therefore, under §6.4.1, the earnings disregard is **$175 per month**."
            )

        result = generate_answer(question, retrieved, client=mock_llm)
        self.assertTrue(result.grounded)
        self.assertIn("6.4.1", result.citations)
        self.assertIn("$175", result.answer)

    def test_temporal_answer_transitional_reporting_5_2(self):
        """Test G: Change occurring before 1 March 2026 retains 10-day period even if determined later."""
        from src.loader import load_full_policy_corpus
        from src.retriever import PolicyRetriever

        corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        retriever = PolicyRetriever(clauses=corpus)

        question = "I moved on 20 February 2026, but the decision is made on 20 March 2026. What reporting deadline applies?"
        retrieved = retriever.retrieve(question, top_k=4)

        def mock_llm(prompt: str) -> str:
            return (
                "Under Amendment No. 2026-01 §5.2, where the change of circumstances occurred before 1 March 2026, "
                "the reporting period is the period that applied at the date of the change, irrespective of the date of determination. "
                "Thus, under §4.3.2, the applicable reporting period is **10 calendar days**."
            )

        result = generate_answer(question, retrieved, client=mock_llm)
        self.assertTrue(result.grounded)
        self.assertIn("4.3.2", result.citations)
        self.assertIn("10 calendar days", result.answer)

    def test_temporal_answer_spanning_period_5_3(self):
        """Test H: Spanning period claim is apportioned under §5.3 and §7.4.3."""
        from src.loader import load_full_policy_corpus
        from src.retriever import PolicyRetriever

        corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        retriever = PolicyRetriever(clauses=corpus)

        question = "How is an award calculated for a claim period from 15 February 2026 to 15 March 2026?"
        retrieved = retriever.retrieve(question, top_k=5)

        def mock_llm(prompt: str) -> str:
            return (
                "Under Amendment No. 2026-01 §5.3, where a claim spans 1 March 2026, the figures in force on each day apply "
                "and the award is apportioned under §7.4.3."
            )

        result = generate_answer(question, retrieved, client=mock_llm)
        self.assertTrue(result.grounded)
        self.assertIn("7.4.3", result.citations)

    def test_temporal_answer_inserted_sanction_exception_10_5_3A(self):
        """Test I: Positive change that increases award cannot be sanctioned under §10.5.3A."""
        from src.loader import load_full_policy_corpus
        from src.retriever import PolicyRetriever

        corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        retriever = PolicyRetriever(clauses=corpus)

        question = "Can a sanction be imposed if an unreported change would have increased the award?"
        retrieved = retriever.retrieve(question, top_k=4)

        def mock_llm(prompt: str) -> str:
            return (
                "Under §10.5.3A (inserted by Amendment No. 2026-01), a sanction must not be imposed in respect of a failure "
                "to report where the change of circumstances in question would have increased the award."
            )

        result = generate_answer(question, retrieved, client=mock_llm)
        self.assertTrue(result.grounded)
        self.assertIn("10.5.3A", result.citations)

    def test_adversarial_1_change_20_feb_det_20_mar(self):
        """Test 1: Change on 20 Feb 2026, determination 20 Mar 2026 -> 10 calendar days under §5.2."""
        from src.loader import load_full_policy_corpus
        from src.retriever import PolicyRetriever

        corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        retriever = PolicyRetriever(clauses=corpus)

        q = "A change occurred on 20 February 2026 and the determination was made on 20 March 2026. What reporting deadline applies?"
        retrieved = retriever.retrieve(q, top_k=4)

        def mock_llm(prompt: str) -> str:
            return (
                "Under Amendment No. 2026-01 §5.2, because the change of circumstances occurred on 20 February 2026 "
                "(before 1 March 2026), the pre-amendment reporting deadline applies regardless of the determination date. "
                "Under §4.3.2, the recipient must report the change within **10 calendar days**."
            )

        result = generate_answer(q, retrieved, client=mock_llm)
        self.assertTrue(result.grounded)
        self.assertIn("4.3.2", result.citations)
        self.assertIn("10 calendar days", result.answer)

    def test_adversarial_2_change_20_mar_det_20_feb(self):
        """Test 2: Change on 20 Mar 2026, determination 20 Feb 2026 -> 14 calendar days under §5.2."""
        from src.loader import load_full_policy_corpus
        from src.retriever import PolicyRetriever

        corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        retriever = PolicyRetriever(clauses=corpus)

        q = "A change occurred on 20 March 2026 and the determination was made on 20 February 2026. What reporting deadline applies?"
        retrieved = retriever.retrieve(q, top_k=4)

        def mock_llm(prompt: str) -> str:
            return (
                "Under Amendment No. 2026-01 §5.2, because the change occurred on 20 March 2026 (on or after 1 March 2026), "
                "the amended deadline applies. Under §4.3.2 as amended, the recipient must report within **14 calendar days**."
            )

        result = generate_answer(q, retrieved, client=mock_llm)
        self.assertTrue(result.grounded)
        self.assertIn("4.3.2", result.citations)
        self.assertIn("14 calendar days", result.answer)

    def test_adversarial_3_claim_jan_det_20_mar(self):
        """Test 3: Claim period January 2026 and determination made on 20 March 2026 -> $175 disregard under §5.1."""
        from src.loader import load_full_policy_corpus
        from src.retriever import PolicyRetriever

        corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        retriever = PolicyRetriever(clauses=corpus)

        q = "The claim period was January 2026 and the determination was made on 20 March 2026. What earnings disregard applies?"
        retrieved = retriever.retrieve(q, top_k=4)

        def mock_llm(prompt: str) -> str:
            return (
                "Under Amendment No. 2026-01 §5.1, amendments apply to any determination made on or after 1 March 2026, "
                "even for a prior period such as January 2026. Therefore, under §6.4.1 as amended, the earnings disregard is **$175 per month**."
            )

        result = generate_answer(q, retrieved, client=mock_llm)
        self.assertTrue(result.grounded)
        self.assertIn("6.4.1", result.citations)
        self.assertIn("$175", result.answer)

    def test_adversarial_4_det_20_feb_claim_jan(self):
        """Test 4: Determination made on 20 February 2026 for January claim -> $120 disregard."""
        from src.loader import load_full_policy_corpus
        from src.retriever import PolicyRetriever

        corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        retriever = PolicyRetriever(clauses=corpus)

        q = "The determination was made on 20 February 2026 for a January 2026 claim. What earnings disregard applies?"
        retrieved = retriever.retrieve(q, top_k=4)

        def mock_llm(prompt: str) -> str:
            return (
                "Because the determination was made on 20 February 2026 (prior to 1 March 2026), under Amendment No. 2026-01 §5.1 "
                "the pre-amendment disregard applies. Under §6.4.1, the earnings disregard is **$120 per month**."
            )

        result = generate_answer(q, retrieved, client=mock_llm)
        self.assertTrue(result.grounded)
        self.assertIn("6.4.1", result.citations)
        self.assertIn("$120", result.answer)

    def test_adversarial_5_unspecified_disregard_both_rules(self):
        """Test 5: Unspecified earnings disregard query generates both historical and current rules."""
        from src.loader import load_full_policy_corpus
        from src.retriever import PolicyRetriever

        corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        retriever = PolicyRetriever(clauses=corpus)

        q = "How much is the earnings disregard?"
        retrieved = retriever.retrieve(q, top_k=4)

        def mock_llm(prompt: str) -> str:
            return (
                "Under the original policy manual (§6.4.1), the first **$120 per month** of earnings is disregarded. "
                "Under Amendment No. 2026-01 §1.1, the disregard is increased to **$175 per month**. "
                "Under Amendment No. 2026-01 §5.1, the $175 disregard applies to any determination made on or after 1 March 2026."
            )

        result = generate_answer(q, retrieved, client=mock_llm)
        self.assertTrue(result.grounded)
        self.assertIn("6.4.1", result.citations)
        self.assertIn("$120", result.answer)
        self.assertIn("$175", result.answer)

    def test_adversarial_6_unspecified_reporting_both_rules(self):
        """Test 6: Unspecified reporting query generates both 10-day and 14-day rules with §5.2 explanation."""
        from src.loader import load_full_policy_corpus
        from src.retriever import PolicyRetriever

        corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        retriever = PolicyRetriever(clauses=corpus)

        q = "What is the reporting deadline?"
        retrieved = retriever.retrieve(q, top_k=4)

        def mock_llm(prompt: str) -> str:
            return (
                "Prior to 1 March 2026, a recipient must report changes within **10 calendar days** under §4.3.2. "
                "Effective 1 March 2026 under Amendment No. 2026-01 §2.1, the reporting period is **14 calendar days**. "
                "Under Amendment No. 2026-01 §5.2, the 14-day rule applies only to changes occurring on or after 1 March 2026."
            )

        result = generate_answer(q, retrieved, client=mock_llm)
        self.assertTrue(result.grounded)
        self.assertIn("4.3.2", result.citations)
        self.assertIn("10 calendar days", result.answer)
        self.assertIn("14 calendar days", result.answer)


    def test_part4_a_valid_original_citation(self):
        """Test Part 4A: Valid original policy citation matching retrieved evidence is accepted."""
        evidence = [self.clause_4_3_2]
        mock_response = "The recipient must report within 10 calendar days [§4.3.2]."

        result = generate_answer(
            question="What is the reporting deadline?",
            retrieved_clauses=evidence,
            client=lambda p: mock_response,
        )

        self.assertTrue(result.grounded)
        self.assertIn("4.3.2", result.citations)
        self.assertEqual(result.unsupported_citations, [])
        self.assertTrue(result.citation_complete)
        self.assertEqual(len(result.validated_citations), 1)
        self.assertEqual(result.validated_citations[0].clause_id, "4.3.2")
        self.assertEqual(result.validated_citations[0].source_document, "policy-manual.md")

    def test_part4_b_unsupported_original_citation(self):
        """Test Part 4B: Citing an original clause NOT in retrieved evidence is rejected as unsupported."""
        evidence = [self.clause_4_3_2]
        mock_response = "The recipient must report within 14 days [§6.4.1]."

        result = generate_answer(
            question="What is the reporting deadline?",
            retrieved_clauses=evidence,
            client=lambda p: mock_response,
        )

        self.assertFalse(result.grounded)
        self.assertIn("6.4.1", result.unsupported_citations)
        self.assertNotIn("6.4.1", result.citations)

    def test_part4_c_valid_amendment_citation(self):
        """Test Part 4C: Citing an amendment provision present in retrieved evidence is accepted."""
        from src.loader import load_amendment
        amendments = load_amendment("data/Amendment No. 2026-01.md")
        amend_5_2 = [c for c in amendments if c.clause_id == "Amendment 2026-01 §5.2"][0]

        evidence = [self.clause_4_3_2, amend_5_2]
        mock_response = "Because the change occurred before 1 March 2026, the previous rule applies [Amendment 2026-01 §5.2]."

        result = generate_answer(
            question="What is the rule?",
            retrieved_clauses=evidence,
            client=lambda p: mock_response,
        )

        self.assertTrue(result.grounded)
        self.assertIn("Amendment 2026-01 §5.2", result.citations)
        self.assertEqual(result.unsupported_citations, [])
        self.assertTrue(result.citation_complete)
        self.assertTrue(any(vc.is_transitional for vc in result.validated_citations))

    def test_part4_d_unsupported_amendment_citation(self):
        """Test Part 4D: Citing an amendment provision NOT in retrieved evidence is rejected."""
        from src.loader import load_amendment
        amendments = load_amendment("data/Amendment No. 2026-01.md")
        amend_5_2 = [c for c in amendments if c.clause_id == "Amendment 2026-01 §5.2"][0]

        evidence = [self.clause_4_3_2, amend_5_2]  
        mock_response = "Because of Amendment 2026-01 §5.1, the new rule applies [Amendment 2026-01 §5.1]."

        result = generate_answer(
            question="What rule applies?",
            retrieved_clauses=evidence,
            client=lambda p: mock_response,
        )

        self.assertFalse(result.grounded)
        self.assertIn("Amendment 2026-01 §5.1", result.unsupported_citations)

    def test_part4_e_no_citation_for_substantive_claim(self):
        """Test Part 4E: Making a substantive factual policy claim with no citations flags missing citations."""
        evidence = [self.clause_4_3_2]
        mock_response = "The reporting deadline is 10 calendar days."  

        result = generate_answer(
            question="What is the reporting deadline?",
            retrieved_clauses=evidence,
            client=lambda p: mock_response,
        )

        self.assertFalse(result.citation_complete, "Substantive claim without citation must fail completeness check")
        self.assertTrue(result.has_missing_citations)

    def test_part4_f_multiple_valid_citations(self):
        """Test Part 4F: Multiple valid citations in single response are all recognized and linked."""
        from src.loader import load_amendment
        amendments = load_amendment("data/Amendment No. 2026-01.md")
        amend_5_2 = [c for c in amendments if c.clause_id == "Amendment 2026-01 §5.2"][0]

        evidence = [self.clause_4_3_2, amend_5_2]
        mock_response = "A recipient must report within 10 calendar days [§4.3.2] under Amendment 2026-01 §5.2."

        result = generate_answer(
            question="What is the reporting deadline?",
            retrieved_clauses=evidence,
            client=lambda p: mock_response,
        )

        self.assertTrue(result.grounded)
        self.assertTrue(result.citation_complete)
        self.assertIn("4.3.2", result.citations)
        self.assertIn("Amendment 2026-01 §5.2", result.citations)
        self.assertEqual(result.unsupported_citations, [])
        self.assertEqual(len(result.validated_citations), 2)
        self.assertEqual(len(result.validated_citations), 2)

    def test_part4_g_mixed_valid_and_invalid_citations(self):
        """Test Part 4G: Response with one valid citation and one invalid citation reports both accurately."""
        evidence = [self.clause_4_3_2]  
        mock_response = "The reporting deadline is 10 days [§4.3.2], and disregard is $175 [§6.4.1]."

        result = generate_answer(
            question="What is the rule?",
            retrieved_clauses=evidence,
            client=lambda p: mock_response,
        )

        self.assertFalse(result.grounded)
        self.assertIn("4.3.2", result.citations)
        self.assertIn("6.4.1", result.unsupported_citations)

    def test_part4_h_numeric_values_not_extracted_as_citations(self):
        """Test Part 4H: Plain numbers, days, currency, and dates are NOT extracted as citations."""
        sample_text = (
            "The recipient must report within 10 calendar days or 14 calendar days from 1 March 2026 "
            "for household earnings of $175 per month with a 15 per cent reduction in 2026."
        )
        extracted = extract_citations(sample_text)
        self.assertEqual(extracted, [], f"Expected 0 extracted citations from plain numeric text, got: {extracted}")

    def test_part4_traceability_metadata_preservation(self):
        """Test Part 4: Validated citations preserve all provenance and clause metadata."""
        from src.loader import load_amendment
        amendments = load_amendment("data/Amendment No. 2026-01.md")
        amend_6_4_1 = [c for c in amendments if c.clause_id == "6.4.1"][0]

        evidence = [amend_6_4_1]
        mock_response = "Under §6.4.1 as amended by Amendment 2026-01 §1.1, the disregard is $175."

        result = generate_answer(
            question="What is the disregard?",
            retrieved_clauses=evidence,
            client=lambda p: mock_response,
        )

        self.assertTrue(result.grounded)
        self.assertGreater(len(result.validated_citations), 0)
        vc = result.validated_citations[0]
        self.assertEqual(vc.clause_id, "6.4.1")
        self.assertTrue(vc.is_amendment)
        self.assertIn("source_document", vc.to_dict())
        self.assertEqual(vc.source_document, "Amendment No. 2026-01.md")



    def test_part5_a_empty_retrieval_fast_refusal(self):
        """Test Part 5A: Empty retrieval results return immediate refusal without invoking LLM."""
        called = False

        def failing_mock(prompt: str) -> str:
            nonlocal called
            called = True
            return "Should not be reached"

        result = generate_answer(
            question="What is the policy on quantum computing?",
            retrieved_clauses=[],
            client=failing_mock,
        )

        self.assertFalse(called, "LLM must not be called when evidence is empty")
        self.assertTrue(result.insufficient_evidence)
        self.assertFalse(result.grounded)
        self.assertEqual(result.citations, [])
        self.assertEqual(result.validated_citations, [])
        self.assertEqual(result.unsupported_citations, [])
        self.assertIn("insufficient", result.answer.lower())

    def test_part5_b_completely_irrelevant_question(self):
        """Test Part 5B: Completely irrelevant questions return zero retrieved clauses and fast refusal."""
        from src.loader import load_full_policy_corpus
        from src.retriever import PolicyRetriever

        corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        retriever = PolicyRetriever(clauses=corpus)

        q = "What is the weather in London today?"
        retrieved = retriever.retrieve(q, top_k=5)
        self.assertEqual(retrieved, [], "Irrelevant query must retrieve 0 clauses")

        result = generate_answer(q, retrieved)
        self.assertTrue(result.insufficient_evidence)
        self.assertFalse(result.grounded)
        self.assertEqual(result.citations, [])

    def test_part5_c_out_of_domain_question(self):
        """Test Part 5C: Out-of-domain general knowledge questions are refused without hallucinating."""
        from src.loader import load_full_policy_corpus
        from src.retriever import PolicyRetriever

        corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        retriever = PolicyRetriever(clauses=corpus)

        out_of_domain_queries = [
            "What is the capital of France?",
            "Who is the current president of the United States?",
            "How do I cook pasta?",
            "Write a Python program to sort a list.",
        ]

        def refusal_mock(prompt: str) -> str:
            return "The provided policy evidence is insufficient to answer this question."

        for q in out_of_domain_queries:
            retrieved = retriever.retrieve(q, top_k=5)
            result = generate_answer(q, retrieved, client=refusal_mock)
            self.assertTrue(result.insufficient_evidence)
            self.assertFalse(result.grounded)
            self.assertEqual(result.citations, [])

    def test_part5_d_unsupported_policy_topic(self):
        """Test Part 5D: If retrieved evidence is unrelated to user question, refusal is returned."""
        evidence = [self.clause_1_4_6]  
        mock_response = "The provided policy evidence is insufficient to answer how to appeal a property tax assessment."

        result = generate_answer(
            question="How do I appeal my property tax?",
            retrieved_clauses=evidence,
            client=lambda prompt: mock_response,
        )

        self.assertTrue(result.insufficient_evidence)
        self.assertFalse(result.grounded)
        self.assertEqual(result.citations, [])
        self.assertEqual(result.unsupported_citations, [])

    def test_part5_e_partial_evidence_strict_grounding(self):
        """Test Part 5E: When evidence supports only part of query, citations are grounded and missing info is not hallucinated."""
        evidence = [self.clause_4_3_2]
        mock_response = (
            "Under §4.3.2, a recipient must report changes within 10 calendar days. "
            "However, the policy evidence does not contain information regarding child care grants."
        )

        result = generate_answer(
            question="What is the reporting deadline and how do I apply for a child care grant?",
            retrieved_clauses=evidence,
            client=lambda prompt: mock_response,
        )

        self.assertTrue(result.grounded)
        self.assertIn("4.3.2", result.citations)
        self.assertEqual(result.unsupported_citations, [])
        self.assertFalse(result.insufficient_evidence)

    def test_part5_f_no_fabricated_citations_in_refusal(self):
        """Test Part 5F: Refusal responses contain zero valid or unsupported citations."""
        evidence = []
        result = generate_answer("Can I get a loan?", evidence)

        self.assertTrue(result.insufficient_evidence)
        self.assertEqual(result.citations, [])
        self.assertEqual(result.validated_citations, [])
        self.assertEqual(result.unsupported_citations, [])

    def test_part5_g_refusal_grounded_safe_contract(self):
        """Test Part 5G: Refusal response adheres strictly to structured AnswerResult contract."""
        result = generate_answer("Random question", [])

        self.assertFalse(result.grounded)
        self.assertTrue(result.insufficient_evidence)
        self.assertTrue(result.citation_complete)
        self.assertFalse(result.has_missing_citations)
        self.assertEqual(result.evidence_clause_ids, [])

        res_dict = result.to_dict()
        self.assertFalse(res_dict["grounded"])
        self.assertTrue(res_dict["insufficient_evidence"])
        self.assertEqual(res_dict["citations"], [])

    def test_part5_h_gemini_invocation_prevention(self):
        """Test Part 5H: Clear zero-evidence cases produce exactly 0 Gemini/LLM invocations."""
        call_count = 0

        def counting_client(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return "Should not execute"

        result = generate_answer("Unanswerable question", [], client=counting_client)
        self.assertEqual(call_count, 0, "LLM must be invoked 0 times when no evidence is retrieved")
        self.assertTrue(result.insufficient_evidence)
        self.assertFalse(result.grounded)

    def test_part5_i_temporal_insufficient_evidence(self):
        """Test Part 5I: Date-specific query lacking required amendment evidence yields refusal."""
        evidence = [self.clause_4_3_2]  
        mock_response = "The provided policy evidence is insufficient to determine post-amendment reporting rules."

        result = generate_answer(
            question="What is the deadline for a change on 15 April 2026?",
            retrieved_clauses=evidence,
            client=lambda prompt: mock_response,
        )

        self.assertTrue(result.insufficient_evidence)
        self.assertFalse(result.grounded)
        self.assertEqual(result.citations, [])


if __name__ == "__main__":
    unittest.main()

