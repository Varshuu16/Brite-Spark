"""
Unit tests for the deterministic policy manual parser.
"""

from pathlib import Path
import unittest

from src.loader import load_policy, get_clause_by_id
from src.models import PolicyClause


class TestPolicyParser(unittest.TestCase):
    """Test suite for loading and parsing the policy manual."""

    @classmethod
    def setUpClass(cls):
        cls.policy_path = Path("data/policy-manual.md")
        cls.clauses = load_policy(cls.policy_path)

    def test_load_policy_file_success(self):
        """Test that policy file loads successfully and returns clauses."""
        self.assertIsInstance(self.clauses, list)
        self.assertGreater(len(self.clauses), 0)
        self.assertEqual(len(self.clauses), 148, "Expected 148 clauses in data/policy-manual.md")

    def test_clause_ids_are_unique(self):
        """Test that every clause ID is unique."""
        clause_ids = [c.clause_id for c in self.clauses]
        self.assertEqual(len(clause_ids), len(set(clause_ids)), "Clause IDs must be strictly unique")

    def test_all_clauses_have_non_empty_content(self):
        """Test that no clause has empty text or missing critical attributes."""
        for clause in self.clauses:
            self.assertTrue(clause.clause_id, "Clause ID must not be empty")
            self.assertTrue(clause.clause_text.strip(), f"Clause {clause.clause_id} text must not be empty")
            self.assertTrue(clause.parent_section, f"Clause {clause.clause_id} must have a parent section")
            self.assertTrue(clause.parent_part, f"Clause {clause.clause_id} must have a parent part")
            self.assertEqual(clause.citation, f"§{clause.clause_id}")

    def test_part_1_definitions_with_titles(self):
        """Test definitions section where clauses have inline titles (e.g. 1.4.1 Applicant)."""
        applicant_clause = get_clause_by_id(self.clauses, "1.4.1")
        self.assertIsNotNone(applicant_clause)
        self.assertEqual(applicant_clause.clause_title, "Applicant")
        self.assertIn("a person who has submitted an application for assistance", applicant_clause.clause_text)

        district_office_clause = get_clause_by_id(self.clauses, "1.4.10")
        self.assertIsNotNone(district_office_clause)
        self.assertEqual(district_office_clause.clause_title, "District office")
        self.assertIn("Calder Central", district_office_clause.clause_text)
        self.assertIn("Northgate", district_office_clause.clause_text)
        self.assertIn("Weybridge", district_office_clause.clause_text)
        self.assertIn("Ash Hill", district_office_clause.clause_text)

    def test_multi_item_and_list_clauses(self):
        """Test clauses with lettered sub-lists (a), (b), etc."""
        clause_2_1_2 = get_clause_by_id(self.clauses, "2.1.2")
        self.assertIsNotNone(clause_2_1_2)
        self.assertIn("(a) is resident in Calder County and satisfies Part 3;", clause_2_1_2.clause_text)
        self.assertIn("(b) is aged 18 or over, or satisfies §2.3;", clause_2_1_2.clause_text)
        self.assertIn("(f) has made a valid application under Part 8.", clause_2_1_2.clause_text)

        clause_4_1_1 = get_clause_by_id(self.clauses, "4.1.1")
        self.assertIsNotNone(clause_4_1_1)
        self.assertIn("(a) is subject to an unexpired sanction under §10.5;", clause_4_1_1.clause_text)
        self.assertIn("(b) is detained in a correctional facility;", clause_4_1_1.clause_text)

    def test_markdown_tables_preserved(self):
        """Test clauses containing markdown tables (e.g., 6.6.1 and 7.2.1)."""
        income_clause = get_clause_by_id(self.clauses, "6.6.1")
        self.assertIsNotNone(income_clause)
        self.assertIn("| Household size | Monthly threshold |", income_clause.clause_text)
        self.assertIn("| 1 | $1,180 |", income_clause.clause_text)
        self.assertIn("| each additional member | + $410 |", income_clause.clause_text)

        needs_clause = get_clause_by_id(self.clauses, "7.2.1")
        self.assertIsNotNone(needs_clause)
        self.assertIn("| Household composition | Monthly needs figure |", needs_clause.clause_text)
        self.assertIn("| Single adult | $1,240 |", needs_clause.clause_text)
        self.assertIn("| Couple | $1,670 |", needs_clause.clause_text)

    def test_bold_numbers_in_body_not_treated_as_clauses(self):
        """Test that inline bold values like **10 calendar days** or **$120 per month** do not split clauses."""
        clause_4_3_2 = get_clause_by_id(self.clauses, "4.3.2")
        self.assertIsNotNone(clause_4_3_2)
        self.assertIn("**10 calendar days**", clause_4_3_2.clause_text)

        clause_6_4_1 = get_clause_by_id(self.clauses, "6.4.1")
        self.assertIsNotNone(clause_6_4_1)
        self.assertIn("**$120 per month**", clause_6_4_1.clause_text)

        clause_9_1_4 = get_clause_by_id(self.clauses, "9.1.4")
        self.assertIsNotNone(clause_9_1_4)
        self.assertIn("**30 calendar days**", clause_9_1_4.clause_text)

    def test_representative_clauses_across_parts(self):
        """Verify representative clauses from every single part of the manual."""
        samples = [
            ("1.1.1", "Part 1 — Scope and Definitions", "1.1 Purpose of the Program"),
            ("2.4.1", "Part 2 — General Conditions of Eligibility", "2.4 Resources"),
            ("3.2.1", "Part 3 — Residence", "3.2 Temporary absence from the County"),
            ("4.3.2", "Part 4 — Exclusions", "4.3 Recipient obligations"),
            ("5.3.1", "Part 5 — Special Household Circumstances", "5.3 Households including a person receiving a training allowance"),
            ("6.1.1", "Part 6 — Income", "6.1 General"),
            ("7.1.1", "Part 7 — Calculation of the Award", "7.1 The award"),
            ("8.3.1", "Part 8 — Applications and Determinations", "8.3 Time limits for determination"),
            ("9.5.1", "Part 9 — Overpayments and Recovery", "9.5 Time limits"),
            ("10.5.2", "Part 10 — Suspension, Termination and Sanctions", "10.5 Sanctions"),
            ("11.1.2", "Part 11 — Review", "11.1 Right to review"),
            ("12.1.1", "Part 12 — Appeal", "12.1 Right of appeal"),
        ]

        for cid, expected_part, expected_sec in samples:
            clause = get_clause_by_id(self.clauses, cid)
            self.assertIsNotNone(clause, f"Clause {cid} should be found")
            self.assertEqual(clause.parent_part, expected_part, f"Clause {cid} parent part mismatch")
            self.assertEqual(clause.parent_section, expected_sec, f"Clause {cid} parent section mismatch")
            self.assertGreater(len(clause.clause_text), 10, f"Clause {cid} text too short")

    def test_invalid_file_raises_error(self):
        """Test that missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_policy("data/non_existent_policy_file.md")


if __name__ == "__main__":
    unittest.main()
