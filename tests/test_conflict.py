"""
Unit tests for deterministic substantive policy conflict detection and handling (Part 6).
"""

import unittest
from typing import List

from src.models import PolicyClause, PolicyCitation, PolicyConflict
from src.loader import load_full_policy_corpus
from src.retriever import PolicyRetriever, ScoredClause
from src.temporal import TemporalContext, TemporalStatus, QueryEventType, extract_temporal_context
from src.conflict import detect_conflicts
from src.answer import generate_answer, AnswerResult


class TestPolicyConflictHandling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_full_policy_corpus("data/policy-manual.md", "data/Amendment No. 2026-01.md")
        cls.retriever = PolicyRetriever(clauses=cls.corpus)

        cls.clause_4_3_2_orig = [c for c in cls.corpus if c.clause_id == "4.3.2" and not c.is_amendment][0]
        cls.clause_4_3_2_amend = [c for c in cls.corpus if c.clause_id == "4.3.2" and c.is_amendment][0]
        cls.clause_9_1_4 = [c for c in cls.corpus if c.clause_id == "9.1.4" and not c.is_amendment][0]
        cls.clause_10_5_1 = [c for c in cls.corpus if c.clause_id == "10.5.1"][0]
        cls.clause_10_5_3A = [c for c in cls.corpus if c.clause_id == "10.5.3A"][0]
        cls.clause_6_4_1_orig = [c for c in cls.corpus if c.clause_id == "6.4.1" and not c.is_amendment][0]
        cls.clause_6_4_1_amend = [c for c in cls.corpus if c.clause_id == "6.4.1" and c.is_amendment][0]

    def test_part6_a_genuine_numeric_conflict(self):
        """Test Part 6A: Substantive reporting deadline discrepancy between §4.3.2 and §9.1.4 is detected."""
        evidence = [self.clause_4_3_2_orig, self.clause_9_1_4]
        conflicts = detect_conflicts(evidence, question="What is the deadline for reporting a change?")

        self.assertEqual(len(conflicts), 1)
        c = conflicts[0]
        self.assertEqual(c.conflict_id, "CONF-REPORTING-01")
        self.assertIn("§4.3.2", c.clause_ids)
        self.assertIn("§9.1.4", c.clause_ids)
        self.assertEqual(c.conflict_type, "SUBSTANTIVE")
        self.assertIn("10 calendar days", c.conflicting_values["§4.3.2"])
        self.assertIn("30 calendar days", c.conflicting_values["§9.1.4"])

    def test_part6_b_genuine_sanction_conflict(self):
        """Test Part 6B: Sanction imposition (§10.5.1) vs statutory prohibition (§10.5.3A) is detected."""
        evidence = [self.clause_10_5_1, self.clause_10_5_3A]
        question = "Can a sanction be applied for an unreported positive change that increases the award?"
        conflicts = detect_conflicts(evidence, question=question)

        self.assertEqual(len(conflicts), 1)
        c = conflicts[0]
        self.assertEqual(c.conflict_id, "CONF-SANCTION-01")
        self.assertEqual(c.conflict_type, "SANCTION_DISCREPANCY")
        self.assertTrue(c.resolution_available)

    def test_part6_c_no_false_positive_for_temporal_amendment(self):
        """Test Part 6C: Original 10-day vs amended 14-day rule with post-amendment date is NOT a conflict."""
        evidence = [self.clause_4_3_2_orig, self.clause_4_3_2_amend]
        q = "What is the reporting deadline for a change occurring on 20 March 2026?"
        t_ctx = extract_temporal_context(q)
        self.assertEqual(t_ctx.status, TemporalStatus.POST_AMENDMENT)

        conflicts = detect_conflicts(evidence, temporal_context=t_ctx, question=q)
        self.assertEqual(conflicts, [], "Post-amendment temporal query must not be flagged as unresolved conflict")

    def test_part6_d_pre_amendment_temporal_case(self):
        """Test Part 6D: Change before 1 March 2026 resolves to 10 days without conflict."""
        evidence = [self.clause_4_3_2_orig, self.clause_4_3_2_amend]
        q = "What is the deadline for a change on 10 February 2026?"
        t_ctx = extract_temporal_context(q)
        self.assertEqual(t_ctx.status, TemporalStatus.PRE_AMENDMENT)

        conflicts = detect_conflicts(evidence, temporal_context=t_ctx, question=q)
        self.assertEqual(conflicts, [], "Pre-amendment query must resolve cleanly without conflict")

    def test_part6_e_post_amendment_temporal_case(self):
        """Test Part 6E: Change after 1 March 2026 resolves to 14 days without conflict."""
        evidence = [self.clause_4_3_2_orig, self.clause_4_3_2_amend]
        q = "What is the deadline for a change on 15 April 2026?"
        t_ctx = extract_temporal_context(q)
        self.assertEqual(t_ctx.status, TemporalStatus.POST_AMENDMENT)

        conflicts = detect_conflicts(evidence, temporal_context=t_ctx, question=q)
        self.assertEqual(conflicts, [], "Post-amendment query must resolve cleanly without conflict")

    def test_part6_f_unspecified_temporal_query_not_a_conflict(self):
        """Test Part 6F: Unspecified temporal query explains both historical & current rules without false conflict."""
        evidence = [self.clause_6_4_1_orig, self.clause_6_4_1_amend]
        q = "What is the earnings disregard?"
        t_ctx = extract_temporal_context(q)
        self.assertEqual(t_ctx.status, TemporalStatus.UNSPECIFIED)

        conflicts = detect_conflicts(evidence, temporal_context=t_ctx, question=q)
        self.assertEqual(conflicts, [], "Unspecified temporal query is chronological versioning, not a conflict")

    def test_part6_g_conflict_citations_fully_grounded(self):
        """Test Part 6G: Answer citing both conflicting clauses is marked grounded and surfaces conflict."""
        evidence = [self.clause_4_3_2_orig, self.clause_9_1_4]
        mock_response = (
            "The retrieved policy evidence contains a discrepancy: under §4.3.2, the recipient reporting deadline "
            "is 10 calendar days, while §9.1.4 refers to a 30 calendar day overpayment notification period."
        )

        result = generate_answer(
            question="What is the deadline for reporting a change?",
            retrieved_clauses=evidence,
            client=lambda prompt: mock_response,
        )

        self.assertTrue(result.grounded)
        self.assertTrue(result.conflicts_detected)
        self.assertEqual(len(result.conflicts), 1)
        self.assertIn("4.3.2", result.citations)
        self.assertIn("9.1.4", result.citations)
        self.assertEqual(result.unsupported_citations, [])
        self.assertFalse(result.insufficient_evidence)

    def test_part6_h_unsupported_conflict_citation_fails_grounding(self):
        """Test Part 6H: Answer citing unretrieved conflict clause §9.9.9 is marked ungrounded."""
        evidence = [self.clause_4_3_2_orig]  
        mock_response = (
            "There is a conflict between §4.3.2 (10 days) and §9.9.9 (30 days)."
        )

        result = generate_answer(
            question="Is there a reporting deadline conflict?",
            retrieved_clauses=evidence,
            client=lambda prompt: mock_response,
        )

        self.assertFalse(result.grounded)
        self.assertIn("9.9.9", result.unsupported_citations)

    def test_part6_i_distinguish_conflict_from_insufficient_evidence(self):
        """Test Part 6I: System cleanly distinguishes genuine conflict from no evidence / refusal."""
        evidence = [self.clause_4_3_2_orig, self.clause_9_1_4]

        res_conflict = generate_answer(
            question="What is the reporting deadline?",
            retrieved_clauses=evidence,
            client=lambda p: "Under §4.3.2 the deadline is 10 days but under §9.1.4 it mentions 30 days.",
        )
        self.assertTrue(res_conflict.conflicts_detected)
        self.assertFalse(res_conflict.insufficient_evidence)
        self.assertTrue(res_conflict.grounded)

        res_refusal = generate_answer(
            question="What is the weather today?",
            retrieved_clauses=[],
        )
        self.assertFalse(res_refusal.conflicts_detected)
        self.assertTrue(res_refusal.insufficient_evidence)
        self.assertFalse(res_refusal.grounded)

    def test_part6_conflict_metadata_serialization(self):
        """Test Part 6: PolicyConflict serialization and AnswerResult dict representation."""
        conflict = PolicyConflict(
            conflict_id="CONF-TEST-01",
            clause_ids=["§1.1", "§1.2"],
            source_documents=["docA.md", "docB.md"],
            conflicting_values={"§1.1": "Value A", "§1.2": "Value B"},
            description="Test conflict",
            conflict_type="SUBSTANTIVE",
        )
        c_dict = conflict.to_dict()
        self.assertEqual(c_dict["conflict_id"], "CONF-TEST-01")
        self.assertEqual(c_dict["clause_ids"], ["§1.1", "§1.2"])
        self.assertEqual(c_dict["conflict_type"], "SUBSTANTIVE")


if __name__ == "__main__":
    unittest.main()
