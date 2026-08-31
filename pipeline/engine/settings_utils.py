# ⛔ READ-ONLY — DO NOT MODIFY. Protected engine module.
"""
settings_utils.py — Field-aware dynamic simulation settings.

Loads field→delay mappings from local CSVs and auto-derives optimal
simulation settings (delay, region, universe) based on which fields
appear in an alpha expression.

This replaces the hardcoded BASE_SETTINGS dicts scattered across all
batch scripts, freeing the system to use ALL available fields from
the CSVs without being buried by a single hardcoded delay value.
"""

import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)

# Base simulation parameters that are universal
DEFAULT_WQ_SETTINGS: Dict[str, Any] = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,             # fallback; overridden per-expression by resolve_settings()
    "decay": 5,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "ON",
    "language": "FASTEXPR",
    "visualization": False,
    "startDate": "2019-01-01",
    "endDate": "2023-12-31",
}

# Common built-in WQ field names that are NOT in any CSV but always available
BUILTIN_FIELDS: Set[str] = {
    "close", "open", "high", "low", "volume", "vwap", "returns", "adv20", "cap",
    "subindustry", "industry", "sector", "market", "none",
}

# ---------------------------------------------------------------------------
# Token extraction from WQ alpha expressions
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


def extract_field_tokens(expression: str) -> Set[str]:
    """Extract potential field identifiers from an alpha expression.

    Strips operators, punctuation, and standalone numeric literals —
    what remains are either WQ datafield IDs or built-in fields.

    Handles field names with embedded digits (e.g. ``anl4_ady_high``,
    ``cashflow_op``, ``debt_lt``) without splitting them.
    """
    # Replace punctuation with spaces — keep "/" for compound field IDs
    for ch in "(),+-*<>!=:":
        expression = expression.replace(ch, " ")

    # Split on whitespace
    raw_tokens = expression.split()

    result: Set[str] = set()
    for t in raw_tokens:
        t = t.strip()
        if not t:
            continue
        # Skip known operators
        if t in _OPERATORS:
            continue
        # Skip standalone numeric literals
        if _NUMERIC_RE.match(t):
            continue
        # Skip grouping constants
        if t in ("subindustry", "industry", "sector", "market", "none"):
            continue
        result.add(t)

    return result


_OPERATORS: Set[str] = {
    "abs", "add", "and", "bucket", "days_from_last_change", "densify",
    "divide", "group_backfill", "group_mean", "group_neutralize", "group_rank",
    "group_scale", "group_zscore", "hump", "if_else", "inverse", "is_nan",
    "kth_element", "last_diff_value", "log", "max", "min", "multiply",
    "normalize", "not", "or", "power", "quantile", "rank", "reverse",
    "scale", "sign", "signed_power", "sqrt", "subtract", "trade_when",
    "ts_arg_max", "ts_arg_min", "ts_av_diff", "ts_backfill", "ts_corr",
    "ts_count_nans", "ts_covariance", "ts_decay_linear", "ts_delay",
    "ts_delta", "ts_mean", "ts_product", "ts_quantile", "ts_rank",
    "ts_regression", "ts_scale", "ts_std_dev", "ts_step", "ts_sum",
    "ts_zscore", "vec_avg", "vec_sum", "winsorize", "zscore",
}

# ---------------------------------------------------------------------------
# Field→delay registry loaded from local CSVs
# ---------------------------------------------------------------------------


class FieldDelayRegistry:
    """Loads field→delay mappings from all CSVs under ``Data and operators/``.

    Cached in memory; call ``load()`` once at startup.
    """

    def __init__(self, csv_dir: Optional[Path] = None) -> None:
        self._csv_dir = csv_dir or Path(__file__).parent.parent / "Data and operators"
        self._field_delay: Dict[str, int] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if not self._csv_dir.exists():
            logger.warning("CSV directory not found: %s — falling back to delay=1 for everything", self._csv_dir)
            self._loaded = True
            return

        found = 0
        for csv_path in sorted(self._csv_dir.glob("*.csv")):
            try:
                with open(csv_path, encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        fid = row.get("id", "").strip()
                        if not fid:
                            continue
                        d_str = row.get("delay", "").strip()
                        try:
                            d = int(d_str)
                        except (ValueError, TypeError):
                            d = 1  # default fallback
                        self._field_delay[fid] = d
                        found += 1
            except Exception as exc:
                logger.warning("Failed to read %s: %s", csv_path, exc)

        logger.info("Loaded %d field→delay mappings from %d CSVs", found, len(list(self._csv_dir.glob("*.csv"))))
        self._loaded = True

    def get_field_delay(self, field_id: str) -> Optional[int]:
        """Return the delay value for a field, or None if unknown."""
        self.load()
        return self._field_delay.get(field_id)

    def resolve_expression_delay(self, expression: str) -> int:
        """Determine the optimal delay for an expression based on its fields.

        Resolution strategy:
          1. Extract all field tokens from the expression.
          2. Look up each known field's delay in the registry.
          3. If ALL known fields share the same delay, use that.
          4. If there's a mix of delay=0 and delay=1 fields, use delay=1
             (delay=0 fields can often still work with delay=1; the reverse is NOT true).
          5. If NO known fields are found, use delay=1 (safe default for price-related fields).
          6. Always fall back to the most common (delay=1) in ambiguous cases.

        Returns:
            ``0`` or ``1``.
        """
        tokens = extract_field_tokens(expression)
        delays_found: Set[int] = set()

        for t in tokens:
            if t in BUILTIN_FIELDS:
                # Built-in fields like close/open/returns are available at both delays
                continue
            d = self.get_field_delay(t)
            if d is not None:
                delays_found.add(d)

        if not delays_found:
            return 1  # safe default
        if len(delays_found) == 1:
            return delays_found.pop()
        # Mix of delays — use 1 as safer common denominator
        return 1

    def resolve_batch_delays(self, expressions: List[str]) -> Dict[str, int]:
        """Resolve delay for each expression in a batch.

        Returns mapping of expression → delay.
        """
        return {expr: self.resolve_expression_delay(expr) for expr in expressions}

    @property
    def all_fields(self) -> Dict[str, int]:
        """Return the full field→delay map."""
        self.load()
        return dict(self._field_delay)

    @property
    def delay0_fields(self) -> List[str]:
        """Return all field IDs that require delay=0."""
        self.load()
        return [fid for fid, d in self._field_delay.items() if d == 0]

    @property
    def delay1_fields(self) -> List[str]:
        """Return all field IDs that work with delay=1."""
        self.load()
        return [fid for fid, d in self._field_delay.items() if d == 1]


# ---------------------------------------------------------------------------
# Convenience: singleton and helpers
# ---------------------------------------------------------------------------

_registry: Optional[FieldDelayRegistry] = None


def get_registry() -> FieldDelayRegistry:
    """Get or create the global FieldDelayRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = FieldDelayRegistry()
    return _registry


def resolve_settings(expression: str,
                     *,
                     overrides: Optional[Dict[str, Any]] = None,
                     base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a simulation settings dict for the given expression.

    Auto-detects delay from the fields in the expression, then applies
    any explicit overrides on top.

    Args:
        expression: WQ alpha expression string.
        overrides: Optional dict of settings to override auto-detected values.
        base: Base settings dict (uses DEFAULT_WQ_SETTINGS if not provided).

    Returns:
        Complete settings dict ready for WQ simulation.
    """
    registry = get_registry()
    settings = dict(base or DEFAULT_WQ_SETTINGS)

    # Auto-detect and set delay
    auto_delay = registry.resolve_expression_delay(expression)
    settings["delay"] = auto_delay

    # Apply explicit overrides
    if overrides:
        settings.update(overrides)

    return settings


def make_batch_item(expression: str,
                    label: Optional[str] = None,
                    *,
                    settings_overrides: Optional[Dict[str, Any]] = None,
                    base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a single batch simulation item with auto-detected settings.

    Returns a dict suitable for ``simulate_batch()``:
    ``{"regular": expr, "settings": {...}, "type": "REGULAR"}``
    """
    settings = resolve_settings(expression, overrides=settings_overrides, base=base)
    return {
        "regular": expression,
        "settings": settings,
        "type": "REGULAR",
    }


def make_batch_items(expressions: List[str],
                     labels: Optional[List[str]] = None,
                     *,
                     settings_overrides: Optional[Dict[str, Any]] = None,
                     base: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Create multiple batch simulation items, each with auto-detected settings."""
    items = []
    for i, expr in enumerate(expressions):
        label = labels[i] if labels and i < len(labels) else f"alpha_{i}"
        items.append(make_batch_item(expr, label, settings_overrides=settings_overrides, base=base))
    return items
