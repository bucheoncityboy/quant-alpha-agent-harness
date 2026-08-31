"""Unit tests for FieldRegistry expression validation (no API calls)."""

import unittest

from engine.field_registry import FieldRegistry, VALID_OPERATORS, COMMON_PARAMS


class TestFieldRegistryValidation(unittest.TestCase):
    def setUp(self):
        # Create FieldRegistry without session for offline validation tests
        self.fr = FieldRegistry(None, cache_path="fields_cache.json")
        # Populate _fields directly for offline testing
        self.fr._fields = {
            "close": {"id": "close", "category": "price_volume"},
            "open": {"id": "open", "category": "price_volume"},
            "high": {"id": "high", "category": "price_volume"},
            "low": {"id": "low", "category": "price_volume"},
            "volume": {"id": "volume", "category": "price_volume"},
            "vwap": {"id": "vwap", "category": "price_volume"},
            "returns": {"id": "returns", "category": "price_volume"},
            "adv20": {"id": "adv20", "category": "price_volume"},
            "cap": {"id": "cap", "category": "fundamental"},
            "equity/cap": {"id": "equity/cap", "category": "value_ratio"},
            "ebit/cap": {"id": "ebit/cap", "category": "value_ratio"},
            "cashflow_op/cap": {"id": "cashflow_op/cap", "category": "value_ratio"},
            "ebit": {"id": "ebit", "category": "profitability"},
            "cashflow_op": {"id": "cashflow_op", "category": "profitability"},
            "enterprise_value": {"id": "enterprise_value", "category": "capital"},
        }

    def test_valid_expression_simple(self):
        """Simple expression with known fields and ops passes validation."""
        is_valid, unknown = self.fr.validate_expression("rank(close)")
        self.assertTrue(is_valid)
        self.assertEqual(unknown, [])

    def test_valid_expression_complex(self):
        """Complex expression with nested operators passes."""
        expr = "group_rank(ts_rank(equity/cap, 60), subindustry)"
        is_valid, unknown = self.fr.validate_expression(expr)
        self.assertTrue(is_valid)
        self.assertEqual(unknown, [])

    def test_valid_expression_with_arithmetic(self):
        """Expression with arithmetic operators passes."""
        expr = "rank(ts_mean(close, 10) - close)"
        is_valid, unknown = self.fr.validate_expression(expr)
        self.assertTrue(is_valid)
        self.assertEqual(unknown, [])

    def test_invalid_field_name(self):
        """Invalid field name is caught."""
        is_valid, unknown = self.fr.validate_expression("rank(sale)")
        self.assertFalse(is_valid)
        self.assertIn("sale", unknown)

    def test_valid_volume_gated_expression(self):
        """Volume-gated expression with trade_when passes."""
        expr = "trade_when(volume > ts_mean(volume, 20), -ts_delta(close, 5), -1)"
        is_valid, unknown = self.fr.validate_expression(expr)
        self.assertTrue(is_valid)
        self.assertEqual(unknown, [])

    def test_valid_value_ratio_expression(self):
        """Value ratio expression passes."""
        expr = "group_rank(ts_rank(ebit/cap, 60), subindustry)"
        is_valid, unknown = self.fr.validate_expression(expr)
        self.assertTrue(is_valid)
        self.assertEqual(unknown, [])

    def test_valid_diversifying_leg(self):
        """Diversifying leg expression passes."""
        expr = "rank(close) + group_zscore(-ts_delta(close, 5), subindustry)"
        is_valid, unknown = self.fr.validate_expression(expr)
        self.assertTrue(is_valid)
        self.assertEqual(unknown, [])

    def test_valid_price_volume_mix(self):
        """Expression with multiple price/volume fields passes."""
        expr = "rank(ts_mean(volume, 20) - ts_mean(volume, 60))"
        is_valid, unknown = self.fr.validate_expression(expr)
        self.assertTrue(is_valid)
        self.assertEqual(unknown, [])

    def test_all_operators_in_valid_set(self):
        """Ensure all critical operators are in VALID_OPERATORS."""
        critical_ops = [
            "rank", "zscore", "group_rank", "group_zscore", "ts_mean",
            "ts_rank", "ts_zscore", "ts_delta", "ts_std_dev", "ts_sum",
            "ts_corr", "trade_when", "if_else", "log", "abs", "sign",
            "sqrt", "inverse", "normalize", "scale", "winsorize",
            "bucket", "quantile", "group_neutralize", "group_mean",
            "group_scale", "group_backfill",
        ]
        for op in critical_ops:
            self.assertIn(op, VALID_OPERATORS, f"{op} missing from VALID_OPERATORS")


if __name__ == "__main__":
    unittest.main()
