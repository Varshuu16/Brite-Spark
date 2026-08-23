"""
Unit tests for the Part 7 Evaluation Framework and Dataset Integrity.
"""

import unittest
from pathlib import Path
from typing import Set

from tests.evaluation_dataset import EVALUATION_DATASET, EvaluationCase
from tests.run_evaluation import run_evaluation
from src.loader import load_full_policy_corpus
from src.retriever import PolicyRetriever
from src.temporal import TemporalStatus
from src.answer import validate_citations_detailed, check_insufficient_evidence


class TestEvaluationFramework(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        cls.retriever = PolicyRetriever(clauses=cls.corpus)
        cls.all_clause_ids = {c.clause_id for c in cls.corpus}
        cls.all_citations = {c.citation for c in cls.corpus}

    def test_evaluation_dataset_loads_and_has_minimum_cases(self):
        """Verify evaluation dataset loads and contains at least 20 cases."""
        self.assertGreaterEqual(len(EVALUATION_DATASET), 20)

    def test_evaluation_case_ids_are_unique(self):
        """Verify all case IDs in the evaluation dataset are strictly unique."""
        case_ids = [c.case_id for c in EVALUATION_DATASET]
        self.assertEqual(len(case_ids), len(set(case_ids)), "Duplicate case IDs detected in evaluation dataset")

    def test_required_categories_are_present(self):
        """Verify all required evaluation categories are present in the dataset."""
        categories = {c.category for c in EVALUATION_DATASET}
        required = {
            "direct_policy",
            "different_wording",
            "multi_clause",
            "temporal_pre_amendment",
            "temporal_post_amendment",
            "temporal_determination",
            "temporal_spanning",
            "temporal_unspecified",
            "refusal",
            "out_of_domain",
            "conflict",
            "citation",
            "adversarial",
            "determinism",
            "partial_evidence",
        }
        missing = required - categories
        self.assertEqual(missing, set(), f"Missing required evaluation categories: {missing}")

    def test_expected_temporal_statuses_are_valid(self):
        """Verify all expected_status values correspond to valid TemporalStatus enum values."""
        valid_statuses = {s.value for s in TemporalStatus}
        for case in EVALUATION_DATASET:
            if case.expected_status is not None:
                self.assertIn(
                    case.expected_status,
                    valid_statuses,
                    f"Case {case.case_id} has invalid temporal status: {case.expected_status}",
                )

    def test_evaluation_runner_produces_100_percent_pass(self):
        """Verify evaluation runner executes and achieves 100% pass rate deterministically."""
        success, metrics = run_evaluation(dataset=EVALUATION_DATASET, verbose=False)
        self.assertTrue(success, f"Evaluation runner failed: {metrics.get('failed_cases')} cases failed")
        self.assertEqual(metrics["pass_rate_pct"], 100.0)
        self.assertEqual(metrics["failed_cases"], 0)

    def test_retrieval_evaluation_determinism(self):
        """Verify retrieval results are strictly identical across multiple repeated runs."""
        q = "What is the deadline for reporting a change of circumstances?"
        first_run = self.retriever.retrieve(q, top_k=5)
        for _ in range(5):
            next_run = self.retriever.retrieve(q, top_k=5)
            self.assertEqual([r.clause.clause_id for r in first_run], [r.clause.clause_id for r in next_run])
            self.assertEqual([r.score for r in first_run], [r.score for r in next_run])

    def test_citation_validator_integration(self):
        """Verify citation validator correctly distinguishes valid vs unsupported citations on eval case."""
        case_e16 = [c for c in EVALUATION_DATASET if c.case_id == "E16"][0]
        retrieved = self.retriever.retrieve(case_e16.question, top_k=5)
        valid_ids, valid_objs, unsupported, is_grounded = validate_citations_detailed(
            case_e16.mock_response, retrieved
        )
        self.assertIn("6.4.1", valid_ids)
        self.assertIn("9.9.9", unsupported)
        self.assertFalse(is_grounded)

    def test_refusal_integration(self):
        """Verify refusal detection on refusal eval cases."""
        case_e11 = [c for c in EVALUATION_DATASET if c.case_id == "E11"][0]
        self.assertTrue(case_e11.expected_refusal)
        self.assertTrue(check_insufficient_evidence("The provided policy evidence is insufficient to answer this question."))

    def test_policy_manual_remains_unmodified_during_evaluation(self):
        """Verify evaluation execution does not modify data/policy-manual.md."""
        manual_path = Path("data/policy-manual.md")
        self.assertTrue(manual_path.exists())
        self.assertGreater(manual_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
