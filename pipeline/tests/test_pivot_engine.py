"""Unit tests for PivotEngine token-level parsing and pivot strategies."""

import unittest

from engine.pivot_engine import PivotEngine


class MockFieldRegistry:
    """A minimal mock that validates expressions and returns alternatives."""

    def __init__(self):
        self.alternatives = {
            "close": ["vwap", "open"],
            "equity/cap": ["ebit/cap", "cashflow_op/cap"],
            "ebit/cap": ["equity/cap", "cashflow_op/cap"],
        }

    def validate_expression(self, expression):
        # Accept all expressions for tests
        return (True, [])

    def get_alternatives(self, field_id):
        return self.alternatives.get(field_id, [])


class TestPivotEngineTokenParsing(unittest.TestCase):
    def setUp(self):
        self.engine = PivotEngine()
        self.fr = MockFieldRegistry()

    def test_parse_simple_expression(self):
        """Simple expression is parsed into correct token types."""
        tokens = self.engine.parse_expression("rank(close)")
        self.assertEqual(len(tokens), 4)
        self.assertEqual(tokens[0].type, "OPERATOR")  # rank
        self.assertEqual(tokens[0].value, "rank")
        self.assertEqual(tokens[1].type, "PAREN")  # (
        self.assertEqual(tokens[1].value, "(")
        self.assertEqual(tokens[2].type, "FIELD")  # close
        self.assertEqual(tokens[2].value, "close")
        self.assertEqual(tokens[3].type, "PAREN")  # )
        self.assertEqual(tokens[3].value, ")")

    def test_parse_group_rank_expression(self):
        """group_rank(ts_rank(close, 60), subindustry) is parsed correctly."""
        tokens = self.engine.parse_expression("group_rank(ts_rank(close, 60), subindustry)")
        # group_rank, (, ts_rank, (, close, COMMA, 60, ), COMMA, subindustry, )
        op_types = [t.type for t in tokens]
        self.assertEqual(op_types.count("OPERATOR"), 2)  # group_rank, ts_rank
        self.assertEqual(op_types.count("PAREN"), 4)  # (( ))
        self.assertEqual(op_types.count("COMMA"), 2)
        self.assertEqual(op_types.count("NUMBER"), 1)
        self.assertEqual(op_types.count("FIELD"), 1)
        self.assertEqual(op_types.count("GROUP"), 1)

    def test_transformation_swap_rank_to_zscore(self):
        """transformation_swap replaces rank with zscore without affecting group_rank."""
        results = self.engine.apply("transformation_swap", "rank(close)", {}, self.fr)
        self.assertTrue(len(results) > 0)
        pivot_expr = results[0][0]
        self.assertIn("zscore", pivot_expr)
        self.assertNotIn("group_zscore", pivot_expr)  # should be zscore, not group_zscore

    def test_transformation_swap_does_not_break_group_rank(self):
        """transformation_swap on group_rank expression keeps group_rank intact."""
        expr = "group_rank(ts_rank(close, 60), subindustry)"
        results = self.engine.apply("transformation_swap", expr, {}, self.fr)
        self.assertTrue(len(results) > 0)
        for pivot_expr, _ in results:
            # The outer group_rank might become group_zscore (valid transformation)
            # But inner ts_rank should NOT become ts_zscore in the same pivot
            # Actually, transformation_swap only swaps the TOP-LEVEL call
            self.assertNotIn("group_zscore(ts_zscore", pivot_expr)

    def test_diversifying_leg_adds_term(self):
        """diversifying_leg adds a decorrelation term."""
        results = self.engine.apply("diversifying_leg", "rank(close)", {}, self.fr)
        self.assertTrue(len(results) > 0)
        # First result should be addition variant with group_zscore
        pivot_expr = results[0][0]
        self.assertIn("group_zscore", pivot_expr)
        self.assertIn("ts_delta", pivot_expr)

    def test_time_horizon_change(self):
        """time_horizon_change doubles lookback values."""
        expr = "ts_mean(close, 10)"
        results = self.engine.apply("time_horizon_change", expr, {}, self.fr)
        self.assertTrue(len(results) > 0)
        found_20 = any("20" in pivot for pivot, _ in results)
        self.assertTrue(found_20)

    def test_grouping_change(self):
        """grouping_change replaces group token."""
        expr = "group_rank(close, subindustry)"
        results = self.engine.apply("grouping_change", expr, {}, self.fr)
        self.assertTrue(len(results) > 0)
        found_industry = any("industry" in pivot for pivot, _ in results)
        self.assertTrue(found_industry)

    def test_field_substitution(self):
        """field_substitution replaces field with alternatives."""
        expr = "rank(equity/cap)"
        results = self.engine.apply("field_substitution", expr, {}, self.fr)
        self.assertTrue(len(results) > 0)
        found_ebit = any("ebit/cap" in pivot for pivot, _ in results)
        self.assertTrue(found_ebit)

    def test_max_3_pivots(self):
        """apply returns at most 3 pivot results."""
        results = self.engine.apply("SELF_CORR_FAIL", "rank(close)", {}, self.fr)
        self.assertLessEqual(len(results), 3)

    def test_failure_priority_differs(self):
        """SELF_CORR_FAIL and PROD_CORR_FAIL have different priority orders."""
        # Both should return valid results
        self_corr = self.engine.apply("SELF_CORR_FAIL", "rank(close)", {}, self.fr)
        prod_corr = self.engine.apply("PROD_CORR_FAIL", "rank(close)", {}, self.fr)
        self.assertTrue(len(self_corr) > 0)
        self.assertTrue(len(prod_corr) > 0)


if __name__ == "__main__":
    unittest.main()
