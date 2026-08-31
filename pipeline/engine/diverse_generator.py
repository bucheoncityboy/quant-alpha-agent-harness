# ⛔ READ-ONLY — DO NOT MODIFY. Protected engine module.
"""Diverse WQ alpha generator using ALL data field categories.

Samples fields from all 8 CSV categories (analyst, fundamental, model, news,
option, pv, sentiment, socialmedia) to build diverse alpha variants.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.settings_utils import make_batch_item, get_registry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "Data and operators"

CATEGORIES = [
    "pv",
    "fundamental",
    "analyst",
    "model",
    "news",
    "option",
    "sentiment",
    "socialmedia",
]

CATEGORY_CSV: Dict[str, Path] = {
    c: DATA_DIR / f"{c}.csv" for c in CATEGORIES
}

GROUPS = ["subindustry", "industry", "sector", "market"]

LOOKBACKS = [5, 10, 15, 20, 21, 30, 40, 45, 60, 63, 90, 120, 252]

DECAYS = [1, 3, 5, 10]

TRUNCATIONS = [0.05, 0.08]

# ---------------------------------------------------------------------------
# Category-aware field loader
# ---------------------------------------------------------------------------


class CategoryFieldLoader:
    """Load fields from all 8 category CSVs, indexed by category."""

    # Fields matching any of these patterns are group identifiers (Unit[Group:1]),
    # incompatible with vector-only operators (ts_*, rank, etc.).
    _GROUP_PATTERNS = (
        "_sector", "_industry", "_subindustry",
        "hierarchy", "_h_",
    )

    def __init__(self) -> None:
        self._cat_fields: Dict[str, List[str]] = {}
        self._cat_delay0: Dict[str, List[str]] = {}
        self._cat_delay1: Dict[str, List[str]] = {}
        self._all_fields: List[str] = []
        self._cat_data_fields: Dict[str, List[str]] = {}  # non-group fields only
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        for cat in CATEGORIES:
            csv_path = CATEGORY_CSV.get(cat)
            if not csv_path or not csv_path.exists():
                continue
            fields, d0, d1 = [], [], []
            data_fields, d0_data = [], []
            with open(csv_path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    fid = row.get("id", "").strip()
                    if not fid:
                        continue
                    fields.append(fid)
                    delay = int(row.get("delay", "1") or 1)
                    if delay == 0:
                        d0.append(fid)
                    else:
                        d1.append(fid)
                    # Separate group-identifier fields from data fields
                    if not self._is_group_field(fid):
                        data_fields.append(fid)
                        if delay == 0:
                            d0_data.append(fid)

            self._cat_fields[cat] = fields
            self._cat_delay0[cat] = d0
            self._cat_delay1[cat] = d1
            self._cat_data_fields[cat] = data_fields
            self._all_fields.extend(fields)
        self._loaded = True

    @staticmethod
    def _is_group_field(fid: str) -> bool:
        """Check if a field ID identifies a group (sector/industry) rather than data.
        Group-level fields return Unit[Group:1] and cannot be used in ts_* operators."""
        lower = fid.lower()
        return any(p in lower for p in CategoryFieldLoader._GROUP_PATTERNS)

    def get_data_fields(self, category: str) -> List[str]:
        """Return non-group data fields for a category (safe for vector/time-series operators)."""
        self.load()
        pool = self._cat_data_fields.get(category, [])
        if not pool:
            pool = self.get_fields(category)  # fallback to all fields
        return pool

    def get_fields(self, category: str) -> List[str]:
        self.load()
        return self._cat_fields.get(category, [])

    def get_delay0_fields(self, category: str) -> List[str]:
        """Return only delay=0 fields from a category (if any)."""
        self.load()
        return self._cat_delay0.get(category, [])

    def get_delay1_fields(self, category: str) -> List[str]:
        """Return only delay=1 fields from a category."""
        self.load()
        return self._cat_delay1.get(category, [])

    def random_field(self, category: str, *, delay: Optional[int] = None) -> str:
        """Pick a random field from a category, optionally filtered by delay."""
        pool = self.get_fields(category)
        if delay == 0:
            pool = self.get_delay0_fields(category)
        elif delay == 1:
            pool = self.get_delay1_fields(category)
        if not pool:
            # Fallback: shift to any field from the category
            pool = self.get_fields(category)
            if not pool:
                return "close"  # ultimate fallback
        return random.choice(pool)

    def random_two_fields(self, category: str, *, delay: Optional[int] = None) -> Tuple[str, str]:
        """Pick two distinct random fields from a category."""
        pool = self.get_fields(category)
        if delay == 0:
            pool = self.get_delay0_fields(category)
        elif delay == 1:
            pool = self.get_delay1_fields(category)
        if len(pool) < 2:
            pool = self.get_fields(category)
            if len(pool) < 2:
                return ("close", "volume")
        return tuple(random.sample(pool, 2))

    @property
    def all_fields(self) -> List[str]:
        self.load()
        return self._all_fields

    def category_summary(self) -> Dict[str, int]:
        self.load()
        return {c: len(f) for c, f in self._cat_fields.items()}

    def delay0_summary(self) -> Dict[str, int]:
        self.load()
        return {c: len(f) for c, f in self._cat_delay0.items()}


# ---------------------------------------------------------------------------
# Field usage cycler — ensures broad coverage per category
# ---------------------------------------------------------------------------


class FieldCycler:
    """Exhaustive per-category field cycler.

    Never repeats a field until every field in that category has been used.
    When all fields are exhausted, resets the category for a new cycle.
    """

    def __init__(self, fl: CategoryFieldLoader) -> None:
        self._fl = fl
        self._ever_used: Dict[str, set] = {}
        self._pool_remaining: Dict[str, List[str]] = {}
        self._global_cycle_count: Dict[str, int] = {}
        for cat in CATEGORIES:
            self._ever_used[cat] = set()
            self._pool_remaining[cat] = []
            self._global_cycle_count[cat] = 0

    def pick(self, category: str, *, delay: Optional[int] = None) -> str:
        """Pick a field from category — NEVER repeats until category is exhausted."""
        pool = self._get_pool(category, delay)
        if not pool:
            return "close"
        # Refill remaining pool if empty (new cycle)
        if not self._pool_remaining[category]:
            self._pool_remaining[category] = [f for f in pool
                                              if f not in self._ever_used[category]]
        # If still empty, all fields have been used — reset for a new cycle
        if not self._pool_remaining[category]:
            self._ever_used[category].clear()
            self._pool_remaining[category] = list(pool)
            self._global_cycle_count[category] += 1
        # Pick from remaining pool (already filtered to unused-in-this-cycle)
        pick = random.choice(self._pool_remaining[category])
        self._pool_remaining[category].remove(pick)
        self._ever_used[category].add(pick)
        return pick

    def pick_two(self, category: str, *, delay: Optional[int] = None) -> Tuple[str, str]:
        """Pick two distinct fields from a category."""
        f1 = self.pick(category, delay=delay)
        f2 = self.pick(category, delay=delay)
        return (f1, f2)

    def pick_three(self, category: str, *, delay: Optional[int] = None) -> Tuple[str, str, str]:
        """Pick three distinct fields from a category."""
        f1 = self.pick(category, delay=delay)
        f2 = self.pick(category, delay=delay)
        f3 = self.pick(category, delay=delay)
        return (f1, f2, f3)

    def lowest_coverage_category(self, delay: Optional[int] = None) -> str:
        """Return the category with the lowest coverage ratio."""
        best_cat = CATEGORIES[0]
        best_pct = 101.0
        for cat in CATEGORIES:
            pool = self._get_pool(cat, delay)
            if not pool:
                continue
            used = len(self._ever_used.get(cat, set()))
            pct = (used / len(pool)) * 100 if pool else 0
            if pct < best_pct:
                best_pct = pct
                best_cat = cat
        return best_cat

    def _get_pool(self, category: str, delay: Optional[int] = None) -> List[str]:
        """Return non-group (data) fields for a category.
        Group-identifier fields (_sector, _industry, hierarchy, etc.) are excluded
        because they return Unit[Group:1] and fail in vector-only operators."""
        self._fl.load()
        if delay == 0:
            return self._fl.get_delay0_fields(category) or self._fl.get_data_fields(category)
        elif delay == 1:
            return self._fl.get_delay1_fields(category) or self._fl.get_data_fields(category)
        return self._fl.get_data_fields(category)

    def coverage_report(self) -> Dict[str, Dict[str, int]]:
        """Return per-category coverage stats (data fields only, excludes group-tier fields)."""
        report = {}
        for cat in CATEGORIES:
            total = len(self._fl.get_data_fields(cat))
            used = len(self._ever_used.get(cat, set()))
            report[cat] = {
                "total": total,
                "used": used,
                "pct": (used / total * 100) if total else 0,
                "cycles": self._global_cycle_count.get(cat, 0),
            }
        return report
# ---------------------------------------------------------------------------
# Expression template library
# ---------------------------------------------------------------------------


def pick_lookback() -> int:
    return random.choice(LOOKBACKS)


def pick_group() -> str:
    return random.choice(GROUPS)


def pick_decay() -> int:
    return random.choice(DECAYS)


def pick_truncation() -> float:
    return random.choice(TRUNCATIONS)


class ExpressionTemplates:
    """Expression templates that take a category field loader and FieldCycler."""

    def __init__(self, fl: CategoryFieldLoader, cycler: Optional[FieldCycler] = None) -> None:
        self._fl = fl
        self._cycler = cycler or FieldCycler(fl)

    def _f(self, cat: str) -> str:
        """Shorthand: pick one field via cycler."""
        return self._cycler.pick(cat)

    def _f2(self, cat: str) -> Tuple[str, str]:
        """Shorthand: pick two distinct fields via cycler."""
        return self._cycler.pick_two(cat)

    def _f3(self, cat: str) -> Tuple[str, str, str]:
        """Shorthand: pick three distinct fields via cycler."""
        return self._cycler.pick_three(cat)

    # ── Existing 14 templates (converted to FieldCycler) ──

    def ts_rank(self, cat: str) -> str:
        return f"ts_rank({self._f(cat)}, {pick_lookback()})"

    def ts_zscore(self, cat: str) -> str:
        return f"ts_zscore({self._f(cat)}, {pick_lookback()})"

    def ts_delta_zscore(self, cat: str) -> str:
        f = self._f(cat)
        d1 = max(pick_lookback() // 2, 5)
        d2 = pick_lookback()
        return f"ts_zscore(ts_delta({f}, {d1}), {d2})"

    def ts_std_dev(self, cat: str) -> str:
        return f"ts_std_dev({self._f(cat)}, {pick_lookback()})"

    def ts_sum(self, cat: str) -> str:
        return f"ts_sum({self._f(cat)}, {pick_lookback()})"

    def rank_ts_mean(self, cat: str) -> str:
        """rank(ts_mean(f, d)) — single-field rank of time-series mean."""
        return f"rank(ts_mean({self._f(cat)}, {pick_lookback()}))"

    def ts_decay_rank(self, cat: str) -> str:
        f = self._f(cat)
        d = random.choice([2, 3, 5, 10, 15, 20, 30, 60])
        return f"ts_decay_linear(rank({f}), {d})"

    def group_rank_ts(self, cat: str) -> str:
        return f"group_rank(ts_rank({self._f(cat)}, {pick_lookback()}), {pick_group()})"

    def group_zscore_ts(self, cat: str) -> str:
        return f"group_zscore(ts_rank({self._f(cat)}, {pick_lookback()}), {pick_group()})"

    def group_zscore_delta(self, cat: str) -> str:
        f = self._f(cat)
        d1 = max(pick_lookback() // 2, 5)
        d2 = pick_lookback()
        return f"group_zscore(ts_delta({f}, {d1}), {pick_group()})"

    def group_neutralize(self, cat: str) -> str:
        return f"group_neutralize(ts_zscore({self._f(cat)}, {pick_lookback()}), {pick_group()})"

    def cross_product(self, cat: str) -> str:
        f1, f2 = self._f2(cat)
        d = pick_lookback()
        return f"rank(ts_mean({f1}, {d})) * rank(ts_mean({f2}, {d}))"

    def ts_corr(self, cat: str) -> str:
        f1, f2 = self._f2(cat)
        return f"ts_corr(rank({f1}), rank({f2}), {pick_lookback()})"

    def hybrid_add(self, cat: str) -> str:
        f1 = self._f(cat)
        f2 = self._f(cat)
        d = pick_lookback()
        return (f"group_rank(ts_rank({f1}, {d}), subindustry) + "
                f"group_zscore(-ts_delta({f2}, {max(d // 2, 5)}), subindustry)")

    def compound_yield(self, cat: str) -> str:
        f = self._f(cat)
        d = pick_lookback()
        return f"group_rank(ts_rank({f} / cap, {d}), {pick_group()})"

    def if_else(self, cat: str) -> str:
        f1 = self._f(cat)
        f2 = self._f(cat)
        d = pick_lookback()
        return f"if_else({f1} > ts_mean({f1}, {d}), rank({f2}), rank(-{f2}))"

    # ── NEW: Time Series templates (8) ──

    def ts_av_diff(self, cat: str) -> str:
        """ts_av_diff(f, d) — difference from moving average."""
        return f"ts_av_diff({self._f(cat)}, {pick_lookback()})"

    def ts_scale_rank(self, cat: str) -> str:
        """ts_scale(rank(f), d) — time-series scale of rank."""
        return f"ts_scale(rank({self._f(cat)}), {pick_lookback()})"

    def ts_product_rank(self, cat: str) -> str:
        """ts_product(rank(f), d) — product of ranked values over window."""
        return f"ts_product(rank({self._f(cat)}), {pick_lookback()})"

    def ts_quantile(self, cat: str) -> str:
        """ts_quantile(f, d) — rank transformed via inverse CDF (Gaussian)."""
        return f"ts_quantile({self._f(cat)}, {pick_lookback()})"

    def ts_ema_rank(self, cat: str) -> str:
        """ts_ema(rank(f), d) — exponential moving average of rank."""
        return f"ts_ema(rank({self._f(cat)}), {pick_lookback()})"

    def ts_argmax_mean(self, cat: str) -> str:
        """ts_arg_max(ts_mean(f, d1), d2) — days since max of moving average."""
        f = self._f(cat)
        d = pick_lookback()
        d2 = max(d // 3, 5)
        return f"ts_arg_max(ts_mean({f}, {d2}), {d})"

    def ts_regression_pair(self, cat: str) -> str:
        """ts_regression(f1, f2, d) — regression slope (rettype=0)."""
        f1, f2 = self._f2(cat)
        return f"ts_regression({f1}, rank({f2}), {pick_lookback()})"

    # ── NEW: Cross Sectional templates (4) ──

    def scale_zscore(self, cat: str) -> str:
        """scale(ts_zscore(f, d)) — scaled z-score for book-size normalization."""
        return f"scale(ts_zscore({self._f(cat)}, {pick_lookback()}))"

    def winsorize_zscore(self, cat: str) -> str:
        """winsorize(ts_zscore(f, d)) — outlier-clipped z-score (std/truncation is a settings parameter, not expression arg)."""
        return f"winsorize(ts_zscore({self._f(cat)}, {pick_lookback()}))"

    def normalize_zscore(self, cat: str) -> str:
        """normalize(ts_zscore(f, d)) — mean-centered z-score."""
        return f"normalize(ts_zscore({self._f(cat)}, {pick_lookback()}))"

    def zscore_rank(self, cat: str) -> str:
        """zscore(rank(f)) — cross-sectional z-score of rank."""
        return f"zscore(rank({self._f(cat)}))"

    # ── NEW: Arithmetic templates (4) ──

    def signed_power_zscore(self, cat: str) -> str:
        """signed_power(ts_zscore(f, d), 2) — signed square of z-score."""
        return f"signed_power(ts_zscore({self._f(cat)}, {pick_lookback()}), 2)"

    def log_rank(self, cat: str) -> str:
        """log(rank(f)) — natural log of rank."""
        return f"log(rank({self._f(cat)}))"

    def sqrt_rank(self, cat: str) -> str:
        """sqrt(rank(f)) — square root of rank."""
        return f"sqrt(rank({self._f(cat)}))"

    def sign_zscore(self, cat: str) -> str:
        """sign(ts_zscore(f, d)) — sign of z-score (+1/0/-1)."""
        return f"sign(ts_zscore({self._f(cat)}, {pick_lookback()}))"

    # ── NEW: Group templates (2) ──

    def group_scale_zscore(self, cat: str) -> str:
        """group_scale(ts_zscore(f, d), group) — group-normalized z-score."""
        return f"group_scale(ts_zscore({self._f(cat)}, {pick_lookback()}), {pick_group()})"

    def group_corr(self, cat: str) -> str:
        """group_neutralize(ts_corr(rank(f1), rank(f2), d), group) — correlation with group neutralize."""
        f1, f2 = self._f2(cat)
        return f"group_neutralize(ts_corr(rank({f1}), rank({f2}), {pick_lookback()}), {pick_group()})"

    # ── NEW: High-complexity templates (3) ──

    def triple_cross(self, cat: str) -> str:
        """3-field cross: rank(ts_mean(f1)) * rank(ts_mean(f2)) + rank(ts_mean(f3))"""
        f1, f2, f3 = self._f3(cat)
        d = pick_lookback()
        return f"rank(ts_mean({f1}, {d})) * rank(ts_mean({f2}, {d})) + rank(ts_mean({f3}, {d}))"

    def layered_corr(self, cat: str) -> str:
        """ts_corr(rank(ts_mean(f1, d1)), rank(ts_mean(f2, d1)), d2) — correlation of ranked means."""
        f1, f2 = self._f2(cat)
        d1 = pick_lookback()
        d2 = max(pick_lookback(), d1 + 5)
        return f"ts_corr(rank(ts_mean({f1}, {d1})), rank(ts_mean({f2}, {d1})), {d2})"

    def if_else_corr(self, cat: str) -> str:
        """if_else(ts_corr(rank(f1), rank(f2), d) > 0, rank(f3), rank(-f3)) — 3-field correlation condition."""
        f1, f2, f3 = self._f3(cat)
        d = pick_lookback()
        return f"if_else(ts_corr(rank({f1}), rank({f2}), {d}) > 0, rank({f3}), rank(-{f3}))"
    # ── FULL COVERAGE: Arithmetic (5) ──

    def abs_zscore(self, cat: str) -> str:
        """abs(ts_zscore(f, d)) — absolute deviation."""
        return f"abs(ts_zscore({self._f(cat)}, {pick_lookback()}))"

    def inverse_rank(self, cat: str) -> str:
        """inverse(rank(f)) — reciprocal of rank."""
        return f"inverse(rank({self._f(cat)}))"

    def power_zscore(self, cat: str) -> str:
        """power(ts_zscore(f, d), 2) — non-signed square of z-score."""
        return f"power(ts_zscore({self._f(cat)}, {pick_lookback()}), 2)"

    def max_pair_zscore(self, cat: str) -> str:
        """max(ts_zscore(f1, d), ts_zscore(f2, d)) — stronger of two signals."""
        f1, f2 = self._f2(cat)
        d = pick_lookback()
        return f"max(ts_zscore({f1}, {d}), ts_zscore({f2}, {d}))"

    def min_pair_zscore(self, cat: str) -> str:
        """min(ts_zscore(f1, d), ts_zscore(f2, d)) — weaker of two signals."""
        f1, f2 = self._f2(cat)
        d = pick_lookback()
        return f"min(ts_zscore({f1}, {d}), ts_zscore({f2}, {d}))"

    # ── FULL COVERAGE: Logical (4) ──

    def is_nan_fill(self, cat: str) -> str:
        """if_else(is_nan(f), ts_mean(f, d), f) — fill NaN with moving average."""
        f = self._f(cat)
        d = pick_lookback()
        return f"if_else(is_nan({f}), ts_mean({f}, {d}), {f})"

    def and_condition(self, cat: str) -> str:
        """if_else(and(cond1, cond2), true_val, false_val) — dual condition."""
        f1, f2, f3 = self._f3(cat)
        d = pick_lookback()
        return f"if_else(and(rank({f1}) > 0, rank({f2}) > 0), rank({f3}), rank(-{f3}))"

    def or_condition(self, cat: str) -> str:
        """if_else(or(cond1, cond2), true_val, false_val) — either condition."""
        f1, f2, f3 = self._f3(cat)
        d = pick_lookback()
        return f"if_else(or(rank({f1}) > 0, rank({f2}) > 0), rank({f3}), rank(-{f3}))"

    def not_condition(self, cat: str) -> str:
        """if_else(not(cond), true_val, false_val) — inverse condition."""
        f = self._f(cat)
        return f"if_else(not(rank({f}) > 0), rank({f}), rank(-{f}))"

    # ── FULL COVERAGE: Time Series (9) ──

    def ts_argmin_mean(self, cat: str) -> str:
        """ts_arg_min(ts_mean(f, d1), d2) — days since min of moving average."""
        f = self._f(cat)
        d = pick_lookback()
        d2 = max(d // 3, 5)
        return f"ts_arg_min(ts_mean({f}, {d2}), {d})"

    def ts_backfill_zscore(self, cat: str) -> str:
        """ts_zscore(ts_backfill(f, 10), d) — backfilled then z-scored."""
        f = self._f(cat)
        d = pick_lookback()
        return f"ts_zscore(ts_backfill({f}, 10), {d})"

    def ts_covariance_pair(self, cat: str) -> str:
        """ts_covariance(rank(f1), rank(f2), d) — covariance of two ranked signals."""
        f1, f2 = self._f2(cat)
        return f"ts_covariance(rank({f1}), rank({f2}), {pick_lookback()})"

    def ts_delay_zscore(self, cat: str) -> str:
        """ts_zscore(ts_delay(f, d1), d2) — delayed value's z-score."""
        f = self._f(cat)
        d1 = pick_lookback()
        d2 = pick_lookback()
        return f"ts_zscore(ts_delay({f}, {d1}), {d2})"

    def ts_count_nans_rank(self, cat: str) -> str:
        """rank(-ts_count_nans(f, d)) — prefer instruments with fewer missing values."""
        f = self._f(cat)
        return f"rank(-ts_count_nans({f}, {pick_lookback()}))"

    def hump_rank(self, cat: str) -> str:
        """hump(rank(f)) — smoothed rank with reduced turnover."""
        return f"hump(rank({self._f(cat)}))"

    def kth_element_value(self, cat: str) -> str:
        """kth_element(f, d, k) — kth value from lookback window."""
        f = self._f(cat)
        d = pick_lookback()
        k = max(d // 4, 2)
        return f"kth_element({f}, {d}, {k})"

    def last_diff_rank(self, cat: str) -> str:
        """last_diff_value(rank(f), d) — most recent different value of rank."""
        f = self._f(cat)
        return f"last_diff_value(rank({f}), {pick_lookback()})"

    def days_from_last_change_rank(self, cat: str) -> str:
        """days_from_last_change(rank(f)) — days since rank last changed."""
        return f"days_from_last_change(rank({self._f(cat)}))"

    # ── FULL COVERAGE: Cross Sectional (1) ──

    def quantile_rank_cs(self, cat: str) -> str:
        """quantile(rank(f), driver='gaussian') — Gaussian-quantiled rank."""
        return f"quantile(rank({self._f(cat)}))"

    # ── FULL COVERAGE: Vector (2) ──

    def vec_avg_rank(self, cat: str) -> str:
        """vec_avg(rank(f)) — mean of rank vector."""
        return f"vec_avg(rank({self._f(cat)}))"

    def vec_sum_rank(self, cat: str) -> str:
        """vec_sum(rank(f)) — sum of rank vector."""
        return f"vec_sum(rank({self._f(cat)}))"

    # ── FULL COVERAGE: Transformational (2) ──

    def bucket_rank(self, cat: str) -> str:
        """bucket(rank(f)) — discretized rank into buckets."""
        return f"bucket(rank({self._f(cat)}))"

    def trade_when_rank(self, cat: str) -> str:
        """trade_when(rank(f), rank(f) > ts_mean(rank(f), d), rank(-f)) — conditional hold."""
        f = self._f(cat)
        d = pick_lookback()
        return f"trade_when(rank({f}), rank({f}) > ts_mean(rank({f}), {d}), rank(-{f}))"

    # ── FULL COVERAGE: Group (2) ──

    def group_backfill_zscore(self, cat: str) -> str:
        """group_backfill(ts_zscore(f, d), group, 20) — group-level NaN fill."""
        f = self._f(cat)
        return f"group_backfill(ts_zscore({f}, {pick_lookback()}), {pick_group()}, 20)"

    def group_mean_rank(self, cat: str) -> str:
        """group_mean(f, rank(f), group) — group-weighted field mean."""
        f = self._f(cat)
        return f"group_mean({f}, rank({f}), {pick_group()})"

    # ── FULL COVERAGE: Time Trend (1) ──

    def ts_step_corr(self, cat: str) -> str:
        """ts_corr(rank(f), ts_step(1), d) — time trend detection via correlation with counter."""
        f = self._f(cat)
        return f"ts_corr(rank({f}), ts_step(1), {pick_lookback()})"

    # ── LAST 3: add, reverse, densify ──

    def add_pair(self, cat: str) -> str:
        """add(rank(f1), rank(f2)) — explicit add() function form (with NaN-as-zero filter)."""
        f1, f2 = self._f2(cat)
        return f"add(rank({f1}), rank({f2}))"

    def reverse_rank(self, cat: str) -> str:
        """reverse(rank(f)) — explicit reverse() = unary minus."""
        return f"reverse(rank({self._f(cat)}))"

    def group_densify(self, cat: str) -> str:
        """group_rank(ts_rank(f, d), densify(subindustry)) — group rank with densified group."""
        f = self._f(cat)
        return f"group_rank(ts_rank({f}, {pick_lookback()}), densify(subindustry))"
    # ── Named template registry ──
    # ── Named template registry (62 total: 16 existing + 20 new + 26 full-coverage) ──
    TEMPLATES: Dict[str, str] = {
        # ── Existing 16 ──
        "ts_rank": "ts_rank(f, d) (low)",
        "ts_zscore": "ts_zscore(f, d) (low)",
        "ts_delta_zscore": "ts_zscore(ts_delta(f)) (med)",
        "ts_std_dev": "ts_std_dev(f, d) (med)",
        "ts_sum": "ts_sum(f, d) (low)",
        "rank_ts_mean": "rank(ts_mean(f, d)) (med)",
        "ts_decay_rank": "ts_decay_linear(rank(f), d) (med)",
        "group_rank_ts": "group_rank(ts_rank(f)) (med)",
        "group_zscore_ts": "group_zscore(ts_rank(f)) (med)",
        "group_zscore_delta": "group_zscore(ts_delta(f)) (med)",
        "group_neutralize": "group_neutralize(ts_zscore(f)) (med)",
        "cross_product": "rank(ts_mean(f1)) * rank(ts_mean(f2)) (high)",
        "ts_corr": "ts_corr(rank(f1), rank(f2), d) (high)",
        "hybrid_add": "group_rank + group_zscore (high)",
        "compound_yield": "group_rank(ts_rank(f/cap)) (med)",
        "if_else": "if_else(cond, rank(f2), rank(-f2)) (high)",
        # ── Previous new 20 ──
        "ts_av_diff": "ts_av_diff(f, d) (low)",
        "ts_scale_rank": "ts_scale(rank(f), d) (med)",
        "ts_product_rank": "ts_product(rank(f), d) (med)",
        "ts_quantile": "ts_quantile(f, d) (med)",
        "ts_ema_rank": "ts_ema(rank(f), d) (med)",
        "ts_argmax_mean": "ts_arg_max(ts_mean(f)) (med)",
        "ts_regression_pair": "ts_regression(f1, rank(f2)) (high)",
        "scale_zscore": "scale(ts_zscore(f)) (med)",
        "winsorize_zscore": "winsorize(ts_zscore(f)) (med)",
        "normalize_zscore": "normalize(ts_zscore(f)) (low)",
        "zscore_rank": "zscore(rank(f)) (low)",
        "signed_power_zscore": "signed_power(ts_zscore(f), 2) (med)",
        "log_rank": "log(rank(f)) (low)",
        "sqrt_rank": "sqrt(rank(f)) (low)",
        "sign_zscore": "sign(ts_zscore(f)) (low)",
        "group_scale_zscore": "group_scale(ts_zscore(f)) (med)",
        "group_corr": "group_neutralize(ts_corr(rank)) (high)",
        "triple_cross": "3-field additive cross (high)",
        "layered_corr": "correlation of ranked means (high)",
        "if_else_corr": "3-field conditional corr (high)",
        # ── Full coverage: Arithmetic (5) ──
        "abs_zscore": "abs(ts_zscore(f, d)) (low)",
        "inverse_rank": "inverse(rank(f)) (low)",
        "power_zscore": "power(ts_zscore(f), 2) (med)",
        "max_pair_zscore": "max(ts_zscore(f1), ts_zscore(f2)) (med)",
        "min_pair_zscore": "min(ts_zscore(f1), ts_zscore(f2)) (med)",
        # ── Full coverage: Logical (4) ──
        "is_nan_fill": "if_else(is_nan(f), ts_mean(f), f) (med)",
        "and_condition": "dual and condition (high)",
        "or_condition": "dual or condition (high)",
        "not_condition": "inverse condition (low)",
        # ── Full coverage: Time Series (9) ──
        "ts_argmin_mean": "ts_arg_min(ts_mean(f)) (med)",
        "ts_backfill_zscore": "ts_zscore(ts_backfill(f)) (med)",
        "ts_covariance_pair": "ts_covariance(rank(f1), rank(f2)) (high)",
        "ts_delay_zscore": "ts_zscore(ts_delay(f)) (med)",
        "ts_count_nans_rank": "rank(-ts_count_nans(f)) (low)",
        "hump_rank": "hump(rank(f)) (low)",
        "kth_element_value": "kth_element(f, d, k) (med)",
        "last_diff_rank": "last_diff_value(rank(f)) (low)",
        "days_from_last_change_rank": "days_from_last_change(rank(f)) (low)",
        # ── Full coverage: Cross Sectional (1) ──
        "quantile_rank_cs": "quantile(rank(f)) (med)",
        # ── Full coverage: Vector (2) ──
        "vec_avg_rank": "vec_avg(rank(f)) (low)",
        "vec_sum_rank": "vec_sum(rank(f)) (low)",
        # ── Full coverage: Transformational (2) ──
        "bucket_rank": "bucket(rank(f)) (low)",
        "trade_when_rank": "trade_when(rank(f)) (high)",
        # ── Full coverage: Group (2) ──
        "group_backfill_zscore": "group_backfill(ts_zscore(f)) (med)",
        "group_mean_rank": "group_mean(f, rank(f)) (high)",
        # ── Full coverage: Time Trend (1) ──
        "ts_step_corr": "ts_corr(rank(f), ts_step(1)) (high)",
        # ── Last 3: add, reverse, densify ──
        "add_pair": "add(rank(f1), rank(f2)) (low)",
        "reverse_rank": "reverse(rank(f)) (low)",
        "group_densify": "group_rank + densify(subindustry) (med)",
    }

    ALL_TEMPLATE_NAMES = list(TEMPLATES.keys())

    def generate(self, category: str, template: str) -> str:
        method = getattr(self, template.replace("-", "_"), None)
        if method is None:
            template = random.choice(self.ALL_TEMPLATE_NAMES)
            method = getattr(self, template)
        return method(category)

    def generate_for_category(self, category: str, n: int = 3) -> List[Dict[str, Any]]:
        """Generate n diverse variants from a single category."""
        fl = self._fl
        fields = fl.get_fields(category)
        if not fields:
            return []
        variants = []
        used_exprs = set()
        for _ in range(n * 3):
            if len(variants) >= n:
                break
            template = random.choice(self.ALL_TEMPLATE_NAMES)
            expr = self.generate(category, template)
            norm = " ".join(expr.split())
            if norm in used_exprs:
                continue
            used_exprs.add(norm)
            label = f"{category[:4]}_{template[:6]}_{pick_lookback()}"
            variants.append({
                "label": label,
                "expr": expr,
                "category": category,
                "template": template,
            })
        return variants

# ---------------------------------------------------------------------------
# Batch generation utilities
# ---------------------------------------------------------------------------


def generate_variants_for_all_categories(
    templates: Optional[List[str]] = None,
    variants_per_category: int = 4,
    delay_filter: Optional[int] = None,
    neutralization_pool: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Generate diverse variants sampling from ALL 8 data categories.

    Args:
        templates: List of template names to use (default: all).
        variants_per_category: How many variants per category.
        delay_filter: If 0, only use delay=0 fields; if 1, only delay=1; None = both.
        neutralization_pool: Pool of neutralization values to cycle through.

    Returns:
        List of dicts with keys: label, expr, neut, category, template.
    """
    fl = CategoryFieldLoader()
    fl.load()
    tmpl = ExpressionTemplates(fl)
    neut_pool = neutralization_pool or GROUPS

    variants = []
    used_exprs = set()

    for cat in CATEGORIES:
        fields = fl.get_fields(cat)
        if not fields:
            continue
        # If delay_filter specified, check whether category has any matching fields
        if delay_filter == 0 and not fl.get_delay0_fields(cat):
            continue
        if delay_filter == 1 and not fl.get_delay1_fields(cat):
            continue

        cat_variants = []
        attempts = 0
        while len(cat_variants) < variants_per_category and attempts < variants_per_category * 5:
            attempts += 1
            template = random.choice(templates or tmpl.ALL_TEMPLATE_NAMES)
            if delay_filter is not None:
                # Force field selection to respect delay filter
                f = fl.random_field(cat, delay=delay_filter)
                # Pick a template that can use a single field
                method = getattr(tmpl, template, None)
                if method is None:
                    continue
                # We need to generate using the specific field
                expr = _generate_with_field(tmpl, template, cat, f, fl)
            else:
                expr = tmpl.generate(cat, template)

            norm = " ".join(expr.split())
            if norm in used_exprs:
                continue
            used_exprs.add(norm)

            neut = random.choice(neut_pool)
            label = f"{cat[:4]}_{template[:6]}_{pick_lookback()}"
            cat_variants.append({
                "label": label,
                "expr": expr,
                "neut": neut,
                "category": cat,
                "template": template,
            })

        variants.extend(cat_variants)

    return variants


def _generate_with_field(
    tmpl: ExpressionTemplates,
    template_name: str,
    category: str,
    field: str,
    fl: CategoryFieldLoader,
) -> str:
    """Generate an expression using a specific field, respecting the template structure."""
    lookback = pick_lookback()
    group = pick_group()
    d1 = max(lookback // 2, 5)

    single_field_templates = {
        # Existing
        "ts_rank": f"ts_rank({field}, {lookback})",
        "ts_zscore": f"ts_zscore({field}, {lookback})",
        "ts_delta_zscore": f"ts_zscore(ts_delta({field}, {d1}), {lookback})",
        "ts_std_dev": f"ts_std_dev({field}, {lookback})",
        "ts_sum": f"ts_sum({field}, {lookback})",
        "rank_ts_mean": f"rank(ts_mean({field}, {lookback}))",
        "ts_decay_rank": f"ts_decay_linear(rank({field}), {random.choice([2,3,5,10,15,20,30,60])})",
        "group_rank_ts": f"group_rank(ts_rank({field}, {lookback}), {group})",
        "group_zscore_ts": f"group_zscore(ts_rank({field}, {lookback}), {group})",
        "group_zscore_delta": f"group_zscore(ts_delta({field}, {d1}), {group})",
        "group_neutralize": f"group_neutralize(ts_zscore({field}, {lookback}), {group})",
        "compound_yield": f"group_rank(ts_rank({field} / cap, {lookback}), {group})",
        # New time series
        "ts_av_diff": f"ts_av_diff({field}, {lookback})",
        "ts_scale_rank": f"ts_scale(rank({field}), {lookback})",
        "ts_product_rank": f"ts_product(rank({field}), {lookback})",
        "ts_quantile": f"ts_quantile({field}, {lookback})",
        "ts_ema_rank": f"ts_ema(rank({field}), {lookback})",
        "ts_argmax_mean": f"ts_arg_max(ts_mean({field}, {d1}), {lookback})",
        # New cross sectional
        "scale_zscore": f"scale(ts_zscore({field}, {lookback}))",
        "winsorize_zscore": f"winsorize(ts_zscore({field}, {lookback}))",
        "normalize_zscore": f"normalize(ts_zscore({field}, {lookback}))",
        "zscore_rank": f"zscore(rank({field}))",
        # New arithmetic
        "signed_power_zscore": f"signed_power(ts_zscore({field}, {lookback}), 2)",
        "log_rank": f"log(rank({field}))",
        "sqrt_rank": f"sqrt(rank({field}))",
        "sign_zscore": f"sign(ts_zscore({field}, {lookback}))",
        # New group
        "group_scale_zscore": f"group_scale(ts_zscore({field}, {lookback}), {group})",
        # ── Full coverage: Arithmetic (3 single-field) ──
        "abs_zscore": f"abs(ts_zscore({field}, {lookback}))",
        "inverse_rank": f"inverse(rank({field}))",
        "power_zscore": f"power(ts_zscore({field}, {lookback}), 2)",
        # ── Full coverage: Logical (2 single-field) ──
        "is_nan_fill": f"if_else(is_nan({field}), ts_mean({field}, {lookback}), {field})",
        "not_condition": f"if_else(not(rank({field}) > 0), rank({field}), rank(-{field}))",
        # ── Full coverage: Time Series (7 single-field) ──
        "ts_argmin_mean": f"ts_arg_min(ts_mean({field}, {d1}), {lookback})",
        "ts_backfill_zscore": f"ts_zscore(ts_backfill({field}, 10), {lookback})",
        "ts_delay_zscore": f"ts_zscore(ts_delay({field}, {d1}), {lookback})",
        "ts_count_nans_rank": f"rank(-ts_count_nans({field}, {lookback}))",
        "hump_rank": f"hump(rank({field}))",
        "kth_element_value": f"kth_element({field}, {lookback}, {max(lookback // 4, 2)})",
        "last_diff_rank": f"last_diff_value(rank({field}), {lookback})",
        "days_from_last_change_rank": f"days_from_last_change(rank({field}))",
        # ── Full coverage: Cross Sectional (1) ──
        "quantile_rank_cs": f"quantile(rank({field}))",
        # ── Full coverage: Vector (2) ──
        "vec_avg_rank": f"vec_avg(rank({field}))",
        "vec_sum_rank": f"vec_sum(rank({field}))",
        # ── Full coverage: Transformational (2) ──
        "bucket_rank": f"bucket(rank({field}))",
        "trade_when_rank": f"trade_when(rank({field}), rank({field}) > ts_mean(rank({field}), {lookback}), rank(-{field}))",
        # ── Full coverage: Group (2) ──
        "group_backfill_zscore": f"group_backfill(ts_zscore({field}, {lookback}), {group}, 20)",
        "group_mean_rank": f"group_mean({field}, rank({field}), {group})",
        # ── Full coverage: Time Trend (1) ──
        "ts_step_corr": f"ts_corr(rank({field}), ts_step(1), {lookback})",
        # ── Last 3 ──
        "reverse_rank": f"reverse(rank({field}))",
    }
    return single_field_templates.get(template_name, f"ts_zscore({field}, {lookback})")


def generate_delay0_batch(
    variants_per_category: int = 4,
) -> List[Dict[str, Any]]:
    """Generate variants using ONLY delay=0 fields across all categories.

    These were previously blocked by the hardcoded delay=1 in BASE_SETTINGS.
    Now they auto-resolve to delay=0 via make_batch_item.
    """
    return generate_variants_for_all_categories(
        templates=[
            "ts_zscore", "ts_delta_zscore", "group_zscore_delta",
            "ts_rank", "ts_av_diff", "ts_quantile",
            "scale_zscore", "winsorize_zscore", "normalize_zscore",
            "zscore_rank", "signed_power_zscore",
            "log_rank", "sqrt_rank", "sign_zscore",
            "group_scale_zscore", "ts_scale_rank", "ts_std_dev", "ts_sum",
        ],
        variants_per_category=variants_per_category,
        delay_filter=0,
    )


def build_batch_items(
    variants: List[Dict[str, Any]],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Convert variant dicts to (label, expr, sim_data) tuples for batch simulation.

    Each sim_data dict is produced by make_batch_item() which auto-resolves delay.
    """
    items = []
    for v in variants:
        overrides = {}
        if v.get("neut"):
            overrides["neutralization"] = v["neut"]
        sd = make_batch_item(v["expr"], settings_overrides=overrides)
        items.append((v["label"], v["expr"], sd))
    return items
