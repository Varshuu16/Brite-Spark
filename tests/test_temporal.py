"""
Unit tests for deterministic temporal parsing and classification.
"""

import datetime
import unittest

from src.temporal import (
    TemporalStatus,
    QueryEventType,
    extract_temporal_context,
    parse_date_string,
)


class TestTemporalEngine(unittest.TestCase):
    """Test suite for deterministic temporal classification and date parsing."""

    def test_pre_amendment_date_extraction(self):
        """Test that dates prior to 1 March 2026 are classified as PRE_AMENDMENT."""
        q1 = "What is the reporting deadline for a change that occurred on 10 February 2026?"
        ctx1 = extract_temporal_context(q1)
        self.assertEqual(ctx1.status, TemporalStatus.PRE_AMENDMENT)
        self.assertEqual(ctx1.event_type, QueryEventType.CHANGE_OF_CIRCUMSTANCES)
        self.assertEqual(ctx1.applicable_transitional_rule, "5.2")
        self.assertEqual(ctx1.detected_date, datetime.date(2026, 2, 10))

        q2 = "How much was the earnings disregard in January 2026?"
        ctx2 = extract_temporal_context(q2)
        self.assertEqual(ctx2.status, TemporalStatus.PRE_AMENDMENT)
        self.assertEqual(ctx2.detected_date, datetime.date(2026, 1, 1))

    def test_post_amendment_date_extraction(self):
        """Test that dates on or after 1 March 2026 are classified as POST_AMENDMENT."""
        q1 = "What is the earnings disregard for a determination made on 15 March 2026?"
        ctx1 = extract_temporal_context(q1)
        self.assertEqual(ctx1.status, TemporalStatus.POST_AMENDMENT)
        self.assertEqual(ctx1.event_type, QueryEventType.DETERMINATION)
        self.assertEqual(ctx1.applicable_transitional_rule, "5.1")
        self.assertEqual(ctx1.detected_date, datetime.date(2026, 3, 15))

        q2 = "What reporting deadline applies for a change in April 2026?"
        ctx2 = extract_temporal_context(q2)
        self.assertEqual(ctx2.status, TemporalStatus.POST_AMENDMENT)
        self.assertEqual(ctx2.applicable_transitional_rule, "5.2")
        self.assertEqual(ctx2.detected_date, datetime.date(2026, 4, 1))

    def test_spanning_period_extraction(self):
        """Test that a claim period spanning 1 March 2026 triggers SPANNING status and §5.3."""
        q = "How is an award calculated for a claim period from 15 February 2026 to 15 March 2026?"
        ctx = extract_temporal_context(q)
        self.assertEqual(ctx.status, TemporalStatus.SPANNING)
        self.assertEqual(ctx.applicable_transitional_rule, "5.3")
        self.assertIsNotNone(ctx.span_start)
        self.assertIsNotNone(ctx.span_end)
        self.assertEqual(ctx.span_start, datetime.date(2026, 2, 15))
        self.assertEqual(ctx.span_end, datetime.date(2026, 3, 15))

    def test_unspecified_date(self):
        """Test that questions with no dates are classified as UNSPECIFIED."""
        q = "What is the deadline for reporting a change of circumstances?"
        ctx = extract_temporal_context(q)
        self.assertEqual(ctx.status, TemporalStatus.UNSPECIFIED)
        self.assertIsNone(ctx.detected_date)

    def test_date_parser_formats(self):
        """Test parser handles ISO, DMY, MDY, and Month-Year strings."""
        self.assertEqual(parse_date_string("2026-02-12"), datetime.date(2026, 2, 12))
        self.assertEqual(parse_date_string("12 February 2026"), datetime.date(2026, 2, 12))
        self.assertEqual(parse_date_string("February 12, 2026"), datetime.date(2026, 2, 12))
        self.assertEqual(parse_date_string("March 2026"), datetime.date(2026, 3, 1))

    def test_adversarial_test1_change_date_controls_pre_amendment(self):
        """Test 1: Mixed dates where change occurred on 20 Feb 2026 and determination was made on 20 Mar 2026."""
        q = "A change occurred on 20 February 2026 and the determination was made on 20 March 2026. What reporting deadline applies?"
        ctx = extract_temporal_context(q)
        self.assertEqual(ctx.status, TemporalStatus.PRE_AMENDMENT)
        self.assertEqual(ctx.applicable_transitional_rule, "5.2")
        self.assertEqual(ctx.controlling_date, datetime.date(2026, 2, 20))
        self.assertEqual(ctx.change_date, datetime.date(2026, 2, 20))
        self.assertEqual(ctx.determination_date, datetime.date(2026, 3, 20))

    def test_adversarial_test2_change_date_controls_post_amendment(self):
        """Test 2: Mixed dates where change occurred on 20 Mar 2026 and determination was made on 20 Feb 2026."""
        q = "A change occurred on 20 March 2026 and the determination was made on 20 February 2026. What reporting deadline applies?"
        ctx = extract_temporal_context(q)
        self.assertEqual(ctx.status, TemporalStatus.POST_AMENDMENT)
        self.assertEqual(ctx.applicable_transitional_rule, "5.2")
        self.assertEqual(ctx.controlling_date, datetime.date(2026, 3, 20))
        self.assertEqual(ctx.change_date, datetime.date(2026, 3, 20))
        self.assertEqual(ctx.determination_date, datetime.date(2026, 2, 20))

    def test_adversarial_test3_determination_date_controls_disregard_post_amendment(self):
        """Test 3: Claim period January 2026 and determination made on 20 March 2026 -> $175 disregard under §5.1."""
        q = "The claim period was January 2026 and the determination was made on 20 March 2026. What earnings disregard applies?"
        ctx = extract_temporal_context(q)
        self.assertEqual(ctx.status, TemporalStatus.POST_AMENDMENT)
        self.assertEqual(ctx.applicable_transitional_rule, "5.1")
        self.assertEqual(ctx.controlling_date, datetime.date(2026, 3, 20))
        self.assertEqual(ctx.determination_date, datetime.date(2026, 3, 20))

    def test_adversarial_test4_determination_date_controls_disregard_pre_amendment(self):
        """Test 4: Determination made on 20 February 2026 for a January 2026 claim -> $120 disregard."""
        q = "The determination was made on 20 February 2026 for a January 2026 claim. What earnings disregard applies?"
        ctx = extract_temporal_context(q)
        self.assertEqual(ctx.status, TemporalStatus.PRE_AMENDMENT)
        self.assertEqual(ctx.applicable_transitional_rule, "5.1")
        self.assertEqual(ctx.controlling_date, datetime.date(2026, 2, 20))
        self.assertEqual(ctx.determination_date, datetime.date(2026, 2, 20))

    def test_adversarial_test5_unspecified_earnings_disregard(self):
        """Test 5: Unspecified date for earnings disregard."""
        q = "How much is the earnings disregard?"
        ctx = extract_temporal_context(q)
        self.assertEqual(ctx.status, TemporalStatus.UNSPECIFIED)
        self.assertIsNone(ctx.controlling_date)

    def test_adversarial_test6_unspecified_reporting_deadline(self):
        """Test 6: Unspecified date for reporting deadline."""
        q = "What is the reporting deadline?"
        ctx = extract_temporal_context(q)
        self.assertEqual(ctx.status, TemporalStatus.UNSPECIFIED)
        self.assertIsNone(ctx.controlling_date)


if __name__ == "__main__":
    unittest.main()
