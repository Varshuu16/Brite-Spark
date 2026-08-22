"""
Unit tests for the deterministic lexical policy retriever.
"""

from pathlib import Path
import unittest

from src.loader import load_policy
from src.retriever import PolicyRetriever, ScoredClause


class TestPolicyRetriever(unittest.TestCase):
    """Test suite for lexical policy retriever functionality and accuracy."""

    @classmethod
    def setUpClass(cls):
        cls.policy_path = Path("data/policy-manual.md")
        cls.clauses = load_policy(cls.policy_path)
        cls.retriever = PolicyRetriever(clauses=cls.clauses)

    def test_direct_match(self):
        """Test A: Direct match query returns top relevant clauses (e.g. reporting deadlines)."""
        results = self.retriever.retrieve("What is the deadline for reporting a change?", top_k=5)
        self.assertGreater(len(results), 0)
        retrieved_ids = [r.clause_id for r in results]
        # Should retrieve §4.3.2 (recipient reporting obligations) or §9.1.4 / §9.1.2
        self.assertTrue(
            "4.3.2" in retrieved_ids or "9.1.4" in retrieved_ids or "9.1.2" in retrieved_ids,
            f"Expected reporting clauses in retrieved IDs: {retrieved_ids}"
        )
        self.assertTrue(all(isinstance(r, ScoredClause) for r in results))
        self.assertGreater(results[0].score, 0.0)

    def test_different_wording(self):
        """Test B: Different wording query retrieving correct conceptual clauses."""
        # Query uses synonyms/phrasings like "job earnings exemption" instead of exact "disregard of employment earnings"
        query = "How much job earnings can a household exempt from countable income?"
        results = self.retriever.retrieve(query, top_k=5)
        self.assertGreater(len(results), 0)
        retrieved_ids = [r.clause_id for r in results]
        # Should retrieve §6.4.1 (Disregards, $120 earnings disregard) or §6.1.1
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
        # Should capture both §2.3.1 (conditions for applicants under 18) and §2.3.2 (supervisor referral)
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
        # Low threshold returns matches
        lenient_results = self.retriever.retrieve(query, top_k=10, min_score=0.1)
        # High threshold only returns high-confidence matches
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


if __name__ == "__main__":
    unittest.main()
