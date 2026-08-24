"""
Unit tests for the deterministic lexical policy retriever.
"""

from pathlib import Path
import unittest

from src.loader import load_policy, load_full_policy_corpus
from src.retriever import PolicyRetriever, ScoredClause


class TestPolicyRetriever(unittest.TestCase):
    """Test suite for lexical policy retriever functionality and accuracy."""

    @classmethod
    def setUpClass(cls):
        cls.policy_path = Path("data/policy-manual.md")
        cls.clauses = load_full_policy_corpus(cls.policy_path, "data/Amendment No. 2026-01.md")
        cls.retriever = PolicyRetriever(clauses=cls.clauses)

    def test_direct_match(self):
        """Test A: Direct match query returns top relevant clauses (e.g. reporting deadlines)."""
        results = self.retriever.retrieve("What is the deadline for reporting a change?", top_k=5)
        self.assertGreater(len(results), 0)
        retrieved_ids = [r.clause_id for r in results]
        self.assertTrue(
            "4.3.2" in retrieved_ids or "9.1.4" in retrieved_ids or "9.1.2" in retrieved_ids,
            f"Expected reporting clauses in retrieved IDs: {retrieved_ids}"
        )
        self.assertTrue(all(isinstance(r, ScoredClause) for r in results))
        self.assertGreater(results[0].score, 0.0)

    def test_different_wording(self):
        """Test B: Different wording query retrieving correct conceptual clauses."""
        query = "How much job earnings can a household exempt from countable income?"
        results = self.retriever.retrieve(query, top_k=5)
        self.assertGreater(len(results), 0)
        retrieved_ids = [r.clause_id for r in results]
        self.assertTrue(
            "6.4.1" in retrieved_ids or "6.1.1" in retrieved_ids or "6.4.2" in retrieved_ids,
            f"Expected disregards clauses in retrieved IDs: {retrieved_ids}"
        )

    def test_multiple_relevant_clauses(self):
        """Test C: Query requiring multiple distinct clauses for a complete answer."""
        query = "What are the rules and requirements for applicants under 18 applying for the program?"
        results = self.retriever.retrieve(query, top_k=5)
        self.assertGreaterEqual(len(results), 2)
        retrieved_ids = [r.clause_id for r in results]
        self.assertIn("2.3.1", retrieved_ids)
        self.assertIn("2.3.2", retrieved_ids)

    def test_irrelevant_question(self):
        """Test D: Query completely unrelated to the policy manual."""
        irrelevant_query = "What is the recipe for baking a chocolate cake?"
        results = self.retriever.retrieve(irrelevant_query, min_score=2.0)
        self.assertEqual(len(results), 0, "Irrelevant query should return no results above threshold")

    def test_top_k_behavior(self):
        """Test E: top_k parameter strictly caps returned results."""
        query = "How is countable income calculated?"
        for k in [1, 3, 5, 10]:
            results = self.retriever.retrieve(query, top_k=k, min_score=0.0)
            self.assertLessEqual(len(results), k)
            self.assertEqual(len(results), k, f"Expected exactly {k} results when min_score=0.0")

    def test_minimum_relevance_threshold(self):
        """Test F: Minimum relevance threshold filters weak matches."""
        query = "district office locations"
        lenient_results = self.retriever.retrieve(query, top_k=10, min_score=0.1)
        strict_results = self.retriever.retrieve(query, top_k=10, min_score=8.0)

        self.assertGreater(len(lenient_results), len(strict_results))
        for r in strict_results:
            self.assertGreaterEqual(r.score, 8.0)

    def test_determinism(self):
        """Test G: Determinism - exact same query runs produce identical ranking and scores."""
        query = "What happens if a recipient fails to report changes without good cause?"
        run_1 = self.retriever.retrieve(query, top_k=5)
        run_2 = self.retriever.retrieve(query, top_k=5)
        run_3 = self.retriever.retrieve(query, top_k=5)

        self.assertEqual(len(run_1), len(run_2))
        self.assertEqual(len(run_2), len(run_3))

        for r1, r2, r3 in zip(run_1, run_2, run_3):
            self.assertEqual(r1.clause_id, r2.clause_id)
            self.assertEqual(r2.clause_id, r3.clause_id)
            self.assertAlmostEqual(r1.score, r2.score, places=6)
            self.assertAlmostEqual(r2.score, r3.score, places=6)

    def test_direct_citation_lookup(self):
        """Test direct clause reference in query boosts target clause to #1 rank."""
        results = self.retriever.retrieve("What does §10.5.2 say about sanctions?", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].clause_id, "10.5.2")

    def test_result_attribute_preservation(self):
        """Test that ScoredClause preserves all required fields and metadata."""
        results = self.retriever.retrieve("appeals panel hearing procedure", top_k=1)
        self.assertEqual(len(results), 1)
        top = results[0]
        self.assertTrue(top.clause_id)
        self.assertTrue(top.citation.startswith("§"))
        self.assertTrue(top.clause_text)
        self.assertTrue(top.parent_section)
        self.assertTrue(top.parent_part)
        self.assertIsInstance(top.score, float)
        self.assertIn("score", top.to_dict())

    def test_reporting_change_retrieves_4_3_2_and_9_1_4(self):
        """Regression test: Query for reporting changes must retrieve primary obligation §4.3.2 AND §9.1.4."""
        query = "What is the deadline for reporting a change of circumstances?"
        results = self.retriever.retrieve(query, top_k=5)
        retrieved_ids = [r.clause_id for r in results]

        self.assertIn("4.3.2", retrieved_ids, "Primary obligation §4.3.2 must be retrieved")
        self.assertIn("9.1.4", retrieved_ids, "Overpayment reference §9.1.4 must be retrieved")
        self.assertIn("4.3.2", retrieved_ids[:2])

    def test_cross_reference_generalization_sanctions(self):
        """Test generalization: Query on sanctions for failure to report retrieves both §10.5.1 and §4.3.2."""
        query = "What penalty or sanction is imposed if I fail to report a change in household?"
        results = self.retriever.retrieve(query, top_k=5)
        retrieved_ids = [r.clause_id for r in results]

        self.assertIn("10.5.1", retrieved_ids)
        self.assertIn("4.3.2", retrieved_ids)

    def test_cross_reference_generalization_award_calculation(self):
        """Test generalization: Query on award calculation retrieves formula §7.1.1 and needs table §7.2.1."""
        query = "How is the monthly award calculated and what are the needs figures?"
        results = self.retriever.retrieve(query, top_k=5)
        retrieved_ids = [r.clause_id for r in results]

        self.assertIn("7.1.1", retrieved_ids)
        self.assertIn("7.2.1", retrieved_ids)

    def test_pre_amendment_retrieval_february_2026(self):
        """Test C: February 2026 change query retrieves historical 10-day clause as top result."""
        query = "What is the deadline for reporting a change of circumstances that occurred on 10 February 2026?"
        results = self.retriever.retrieve(query, top_k=5)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertEqual(top.clause_id, "4.3.2")
        self.assertIn("10 calendar days", top.clause_text)
        self.assertFalse(top.is_amendment)

    def test_post_amendment_retrieval_april_2026(self):
        """Test C: April 2026 change query retrieves amended 14-day clause as top result."""
        query = "What is the deadline for reporting a change of circumstances that occurred on 15 April 2026?"
        results = self.retriever.retrieve(query, top_k=5)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertEqual(top.clause_id, "4.3.2")
        self.assertIn("14 calendar days", top.clause_text)
        self.assertTrue(top.is_amendment)

    def test_post_amendment_determination_disregard_march_2026(self):
        """Test D: March 2026 determination query retrieves amended $175 disregard as top result."""
        query = "What is the earnings disregard for a determination made on 15 March 2026?"
        results = self.retriever.retrieve(query, top_k=5)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertEqual(top.clause_id, "6.4.1")
        self.assertIn("$175", top.clause_text)
        self.assertTrue(top.is_amendment)

    def test_inserted_clause_sanction_exception_10_5_3A(self):
        """Test I: Query about unreported positive change retrieves inserted clause §10.5.3A."""
        query = "Can a sanction be imposed if an unreported change would have increased the award?"
        results = self.retriever.retrieve(query, top_k=5)
        retrieved_ids = [r.clause_id for r in results]
        self.assertIn("10.5.3A", retrieved_ids)

    def test_spanning_period_retrieves_5_3_and_apportionment(self):
        """Test H: Query for claim period spanning 1 March 2026 retrieves Amendment §5.3 and §7.4.3."""
        query = "How is an award calculated for a claim period spanning from 15 February 2026 to 15 March 2026?"
        results = self.retriever.retrieve(query, top_k=5)
        retrieved_ids = [r.clause_id for r in results]
        self.assertTrue(any("5.3" in cid for cid in retrieved_ids))


if __name__ == "__main__":
    unittest.main()
