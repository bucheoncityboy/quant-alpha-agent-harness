"""Tests for engine/alpha_researcher.py — pure unit tests, no WQ API calls."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from engine.alpha_researcher import (
    OperatorDef,
    OperatorNode,
    OperatorRegistry,
    OperatorTreeCompiler,

    PatternTracker,
    FeedbackSample,
    FeedbackStore,
    ValidateGate,
    GROUPS,
    LOOKBACKS,
    _INFIX_SYMBOLS,
    _build_operator_defs,
)

# ===================================================================
# Helpers
# ===================================================================




_VALID_CATEGORIES = {
    "arithmetic", "logical", "time_series", "cross_sectional",
    "vector", "transformational", "group",
}

# ===================================================================
# 1. OperatorRegistry tests
# ===================================================================


class TestOperatorRegistry:
    """OperatorRegistry — all 66 operators, category queries, required fields."""

    def test_registry_loads_66_operators(self):
        """OperatorRegistry loads all 66 operators, .count == 66."""
        registry = OperatorRegistry(operators_path="/nonexistent/operators.md")
        assert registry.count == 66, f"Expected 66 operators, got {registry.count}"

    def test_registry_get_by_category(self):
        """Each of the 7 categories has at least 1 operator."""
        registry = OperatorRegistry(operators_path="/nonexistent/operators.md")
        for cat in _VALID_CATEGORIES:
            ops = registry.get_by_category(cat)
            assert len(ops) >= 1, f"Category '{cat}' has 0 operators"

    def test_registry_known_operators(self):
        """Specific operators exist: ts_rank, ts_zscore, rank, group_rank, ts_corr, add, if_else."""
        registry = OperatorRegistry(operators_path="/nonexistent/operators.md")
        for name in ("ts_rank", "ts_zscore", "rank", "group_rank", "ts_corr", "add", "if_else"):
            assert registry.get(name) is not None, f"Operator '{name}' not found"

    def test_operator_def_has_required_fields(self):
        """Each OperatorDef has name, arity, category, input_types, output_type."""
        registry = OperatorRegistry(operators_path="/nonexistent/operators.md")
        for op in registry.list_all():
            assert isinstance(op.name, str) and op.name, f"Missing name in {op}"
            assert isinstance(op.arity, tuple) and len(op.arity) == 2, (
                f"Bad arity in '{op.name}'"
            )
            assert isinstance(op.category, str) and op.category, (
                f"Missing category in '{op.name}'"
            )
            assert isinstance(op.input_types, list) and len(op.input_types) >= 0, (
                f"Missing input_types in '{op.name}'"
            )
            assert isinstance(op.output_type, str) and op.output_type, (
                f"Missing output_type in '{op.name}'"
            )


# ===================================================================
# 3. OperatorTreeCompiler tests
# ===================================================================


class TestOperatorTreeCompiler:
    """OperatorTreeCompiler — compile/validate trees."""

    def _make_registry(self) -> OperatorRegistry:
        return OperatorRegistry(operators_path="/nonexistent/operators.md")

    def test_compile_simple_unary(self):
        """compile(OperatorNode('abs', ['close'])) → 'abs(close)'."""
        registry = self._make_registry()
        compiler = OperatorTreeCompiler(registry)
        node = OperatorNode("abs", ["close"])
        result = compiler.compile(node)
        assert result == "abs(close)", f"Expected 'abs(close)', got '{result}'"

    def test_compile_binary(self):
        """compile nested → 'ts_corr(rank(close), rank(volume), 40)'."""
        registry = self._make_registry()
        compiler = OperatorTreeCompiler(registry)
        node = OperatorNode("ts_corr", [
            OperatorNode("rank", ["close"]),
            OperatorNode("rank", ["volume"]),
            "40",
        ])
        result = compiler.compile(node)
        assert result == "ts_corr(rank(close), rank(volume), 40)", (
            f"Unexpected: '{result}'"
        )

    def test_compile_with_lookback(self):
        """compile(OperatorNode('ts_mean', ['close', '20'])) → 'ts_mean(close, 20)'."""
        registry = self._make_registry()
        compiler = OperatorTreeCompiler(registry)
        node = OperatorNode("ts_mean", ["close", "20"])
        result = compiler.compile(node)
        assert result == "ts_mean(close, 20)", f"Unexpected: '{result}'"

    def test_validate_valid_tree(self):
        """validate returns empty list for a valid tree."""
        registry = self._make_registry()
        compiler = OperatorTreeCompiler(registry)
        node = OperatorNode("ts_corr", [
            OperatorNode("rank", ["close"]),
            OperatorNode("rank", ["volume"]),
            "40",
        ])
        errors = compiler.validate(node)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_validate_invalid_arity(self):
        """validate returns error for wrong argument count."""
        registry = self._make_registry()
        compiler = OperatorTreeCompiler(registry)
        # ts_corr expects 3 args (vector, vector, lookback) — provide 1
        node = OperatorNode("ts_corr", ["close"])
        errors = compiler.validate(node)
        assert len(errors) >= 1, "Expected at least 1 arity error"
        assert any("expected at least" in e for e in errors), (
            f"No arity error in: {errors}"
        )

    def test_validate_unknown_operator(self):
        """validate returns error for an unknown operator name."""
        registry = self._make_registry()
        compiler = OperatorTreeCompiler(registry)
        node = OperatorNode("nonexistent_op", ["close"])
        errors = compiler.validate(node)
        assert len(errors) >= 1, "Expected validation error"
        assert any("Unknown operator" in e for e in errors), (
            f"No unknown-operator error: {errors}"
        )


# ===================================================================
# 4. PatternTracker tests
# ===================================================================


class TestPatternTracker:
    """PatternTracker — 5-strike rule, blacklist, reset."""

    def test_tracker_starts_clean(self):
        """No blacklisted skeletons initially."""
        tracker = PatternTracker()
        assert len(tracker.blacklisted_skeletons()) == 0
        assert tracker.must_switch() is False

    def test_tracker_5_failures_blacklists(self):
        """5 consecutive failures with fitness < 0.5 blacklists the skeleton."""
        tracker = PatternTracker()
        tracker.FITNESS_THRESHOLD = 0.5
        expr = "ts_zscore(close, 21)"
        for i in range(5):
            tracker.record_result(expr, 0.3)
        assert len(tracker.blacklisted_skeletons()) >= 1, (
            "Skeleton not blacklisted after 5 failures"
        )
        assert tracker.must_switch() is True, "must_switch() should be True"

    def test_tracker_success_resets_counter(self):
        """One success between failures resets the counter."""
        tracker = PatternTracker()
        tracker.FITNESS_THRESHOLD = 0.5
        expr = "ts_zscore(close, 21)"
        # 3 failures
        for _ in range(3):
            tracker.record_result(expr, 0.3)
        # 1 success (fitness >= threshold)
        tracker.record_result(expr, 0.7)
        # Should not be blacklisted yet — need 5 consecutive
        for _ in range(3):
            tracker.record_result(expr, 0.3)
        assert len(tracker.blacklisted_skeletons()) == 0, (
            "Counter was not reset by success"
        )
        # 2 more failures (total 5 after reset) → blacklisted
        for _ in range(2):
            tracker.record_result(expr, 0.3)
        assert len(tracker.blacklisted_skeletons()) >= 1, (
            "Not blacklisted after 5 consecutive failures post-reset"
        )

    def test_tracker_must_switch(self):
        """must_switch() returns True only after >=5 consecutive failures."""
        tracker = PatternTracker()
        tracker.FITNESS_THRESHOLD = 0.5
        expr = "ts_zscore(close, 21)"
        # 4 failures → must_switch should be False
        for _ in range(4):
            tracker.record_result(expr, 0.3)
        assert tracker.must_switch() is False, (
            "must_switch() premature after 4 failures"
        )
        # 5th failure → must_switch True
        tracker.record_result(expr, 0.3)
        assert tracker.must_switch() is True, (
            "must_switch() still False after 5 failures"
        )

    def test_tracker_strike_count(self):
        """strike_count() returns consecutive failure count for a skeleton."""
        tracker = PatternTracker()
        tracker.FITNESS_THRESHOLD = 0.5
        expr = "rank(close)"
        assert tracker.strike_count(expr) == 0
        for _ in range(3):
            tracker.record_result(expr, 0.3)
        assert tracker.strike_count(expr) == 3, (
            f"Expected strike count 3, got {tracker.strike_count(expr)}"
        )


# ===================================================================
# 5. FeedbackStore tests
# ===================================================================


class TestFeedbackStore:
    """FeedbackStore — add/read, aggregates, max samples."""

    def test_store_add_and_read(self, tmp_path):
        """Add a FeedbackSample, then read it back."""
        path = str(tmp_path / "test_feedback.json")
        store = FeedbackStore(path)
        sample = FeedbackSample(
            expression="ts_zscore(close, 20)",
            field_ids=["close"],
            fitness=0.45,
            sharpe=1.2,
            gate1_status="PASS",
            gate2_passed=True,
            verify_5_passed=False,
        )
        store.add_sample(sample)
        # Read via get_context
        ctx = store.get_context()
        assert len(ctx["last_n_results"]) == 1
        last = ctx["last_n_results"][0]
        assert last["expression"] == "ts_zscore(close, 20)"
        assert last["fitness"] == 0.45
        assert last["sharpe"] == 1.2
        assert last["gate1_status"] == "PASS"

    def test_store_context_has_aggregates(self, tmp_path):
        """get_context() returns dict with per-operator aggregates."""
        path = str(tmp_path / "test_feedback_agg.json")
        store = FeedbackStore(path)

        samples = [
            FeedbackSample(
                expression="ts_zscore(close, 20)",
                field_ids=["close"],
                fitness=0.5,
                sharpe=1.0,
            ),
            FeedbackSample(
                expression="ts_zscore(close, 20)",
                field_ids=["close"],
                fitness=0.7,
                sharpe=1.5,
            ),
            FeedbackSample(
                expression="rank(volume)",
                field_ids=["volume"],
                fitness=0.3,
                sharpe=0.8,
            ),
        ]
        for s in samples:
            store.add_sample(s)

        ctx = store.get_context()
        assert "per_operator_stats" in ctx, "Missing per_operator_stats"
        stats = ctx["per_operator_stats"]
        # ts_zscore appears twice
        assert "ts_zscore" in stats, "ts_zscore missing from stats"
        assert stats["ts_zscore"]["count"] == 2
        # avg_fitness = (0.5 + 0.7) / 2 = 0.6
        assert stats["ts_zscore"]["avg_fitness"] == pytest.approx(0.6, abs=1e-4)
        # rank appears once
        assert "rank" in stats
        assert stats["rank"]["count"] == 1

    def test_store_max_samples(self, tmp_path):
        """Adding 101 samples keeps only latest 100 in context."""
        path = str(tmp_path / "test_feedback_max.json")
        store = FeedbackStore(path)

        for i in range(101):
            sample = FeedbackSample(
                expression=f"abs(close)",
                field_ids=["close"],
                fitness=0.1 * (i % 10),
            )
            store.add_sample(sample)

        ctx = store.get_context()
        # get_context uses self._samples[-100:] for last_n_results
        last_n = ctx["last_n_results"]
        assert len(last_n) == 100, (
            f"Expected 100 results, got {len(last_n)}"
        )
        # The oldest entry should be from index 1 (not index 0)
        # The latest one should match the last entry
        assert last_n[-1]["fitness"] == 0.0, (
            "Last entry should have fitness from i=100 (100 % 10 = 0)"
        )

    def test_store_summary(self, tmp_path):
        """get_summary() returns correct aggregates."""
        path = str(tmp_path / "test_feedback_summary.json")
        store = FeedbackStore(path)

        samples = [
            FeedbackSample(expression="abs(close)", field_ids=["close"],
                           fitness=0.5, gate1_status="PASS", gate2_passed=True),
            FeedbackSample(expression="rank(close)", field_ids=["close"],
                           fitness=0.0, gate1_status="FAIL", gate2_passed=False),
        ]
        for s in samples:
            store.add_sample(s)

        summary = store.get_summary()
        assert summary["total_samples"] == 2
        assert summary["gate1_pass_rate"] == 0.5
        assert summary["gate2_pass_rate"] == 0.5
        assert summary["avg_fitness"] == 0.25


# ===================================================================
# 6. ValidateGate tests
# ===================================================================


class TestValidateGate:
    """ValidateGate — token validation against operator registry."""

    def _make_registry(self) -> OperatorRegistry:
        return OperatorRegistry(operators_path="/nonexistent/operators.md")

    def test_validate_known_expression(self):
        """"ts_zscore(close, 20)" returns (True, [])."""
        registry = self._make_registry()
        valid, invalid = ValidateGate.check("ts_zscore(close, 20)", registry)
        assert valid is True, f"Expected valid, got invalid tokens: {invalid}"
        assert invalid == [], f"Expected no invalid tokens, got: {invalid}"

    def test_validate_unknown_token(self):
        """"unknown_operator(close)" returns (False, list) with the unknown token."""
        registry = self._make_registry()
        valid, invalid = ValidateGate.check("unknown_operator(close)", registry)
        assert valid is False, "Expected invalid"
        assert len(invalid) >= 1, "Expected at least 1 invalid token"
        assert "unknown_operator" in invalid, (
            f"'unknown_operator' not in invalid tokens: {invalid}"
        )

    def test_validate_partial_unknown(self):
        """Expression with some known and some unknown tokens detected."""
        registry = self._make_registry()
        # "add" is known, "mystery_op" is not
        valid, invalid = ValidateGate.check("add(close, mystery_op)", registry)
        assert valid is False, "Expected invalid due to mystery_op"
        assert "mystery_op" in invalid, (
            f"'mystery_op' not in invalid tokens: {invalid}"
        )

    def test_validate_infix_expression(self):
        """Infix expressions like 'close > open' are valid."""
        registry = self._make_registry()
        valid, invalid = ValidateGate.check("close > open", registry)
        assert valid is True, f"Expected valid infix, got: {invalid}"

    def test_validate_numeric_tokens(self):
        """Numeric literals are valid tokens."""
        registry = self._make_registry()
        valid, invalid = ValidateGate.check("ts_zscore(close, 20.5)", registry)
        assert valid is True, f"Numeric literal not valid: {invalid}"


# ===================================================================
# 7. Hardcoded operator definitions sanity
# ===================================================================


class TestOperatorDefs:
    """Sanity checks on _build_operator_defs internals."""

    def test_operator_defs_has_all_categories(self):
        """All 7 categories have the correct operators."""
        defs = _build_operator_defs()
        cats: Dict[str, int] = {}
        for op in defs.values():
            cats[op.category] = cats.get(op.category, 0) + 1

        assert cats.get("arithmetic", 0) == 15
        assert cats.get("logical", 0) == 11  # 5 named + 6 infix
        assert cats.get("time_series", 0) == 24
        assert cats.get("cross_sectional", 0) == 6
        assert cats.get("vector", 0) == 2
        assert cats.get("transformational", 0) == 2
        assert cats.get("group", 0) == 6
        assert sum(cats.values()) == 66

    def test_infix_symbols_match_defs(self):
        """Every infix symbol has a corresponding OperatorDef."""
        defs = _build_operator_defs()
        for sym in _INFIX_SYMBOLS:
            if sym in "+-*/":
                # +, -, *, / are not in the 66 (they're infix math, not logical)
                continue
            assert sym in defs, f"Infix symbol '{sym}' missing from operator defs"
