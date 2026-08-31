# ⛔ READ-ONLY — DO NOT MODIFY. Protected engine module.
"""
MyAlphaPool — Existing alpha pool with 2-level dedup + correlation reference.

Two data sources:
  1. my_alphas_cache.json (1,254 test/attempted alphas from WQ API)
     - Used for: EXACT + STRUCTURAL dedup (is_duplicate)
  2. submitted_alphas.json (14 final submitted alphas)
     - Used for: EXACT + STRUCTURAL dedup (is_duplicate)
     - Used for: CORRELATION REFERENCE (pool.submitted)

Detection levels:
  1. exact_duplicate  — identical expression string (against BOTH sources)
  2. structural_duplicate — same operator skeleton, different fields (against BOTH)
  3. new_alpha        — different family + different core fields

Also provides pivot suggestions for structural duplicates and
operator skeleton extraction for expression signatures.
"""

import json
import logging
import os
import re


# Valid WQ statuses that indicate a truly submitted (ACTIVE) alpha.
# "submitted" = POST /alphas/{id}/submit completed; alpha is live in the competition.
# Alphas with other statuses (draft, simulated, check-passed, etc.)
# have NOT been submitted and must NOT be treated as submitted alphas
# for correlation reference or reporting.
VALID_SUBMITTED_STATUSES = frozenset({"submitted", "active"})
from typing import Any, Dict, List, Optional, Tuple

from . import ace_lib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known operators (from operators.md) — used to distinguish operators from
# field names during structural pattern extraction.
# ---------------------------------------------------------------------------
VALID_OPERATORS = frozenset({
    # Arithmetic
    "abs", "add", "densify", "divide", "inverse", "log", "max", "min",
    "multiply", "power", "reverse", "sign", "signed_power", "sqrt",
    "subtract",
    # Cross-sectional
    "bucket", "group_backfill", "group_mean", "group_neutralize",
    "group_rank", "group_scale", "group_zscore", "normalize", "quantile",
    "rank", "scale", "winsorize", "zscore",
    # Time-series
    "ts_arg_max", "ts_arg_min", "ts_av_diff", "ts_backfill", "ts_corr",
    "ts_decay_linear", "ts_delta", "ts_delay", "ts_mean", "ts_product",
    "ts_quantile", "ts_rank", "ts_regression", "ts_scale", "ts_std_dev",
    "ts_sum", "ts_zscore",
    # Logical / conditional
    "if_else", "is_nan", "not", "and", "or", "trade_when",
    # Vector
    "vec_avg", "vec_sum",
    # Special
    "last_diff_value", "days_from_last_change", "kth_element",
    "ts_count_nans", "ts_covariance", "ts_step",
    # Common group/param tokens (not operators but not fields)
    "subindustry", "industry", "sector", "market", "none",
    # Common price/volume fields (frequently used, not operators)
    "cap", "close", "open", "high", "low", "volume", "vwap", "returns",
    "adv20",
})

# Operators that take a group parameter as their last argument
GROUP_OPERATORS = frozenset({
    "group_rank", "group_zscore", "group_mean", "group_neutralize",
    "group_scale", "group_backfill",
})

# Operators that are time-series (take a lookback number)
TS_OPERATORS = frozenset({
    "ts_arg_max", "ts_arg_min", "ts_av_diff", "ts_backfill", "ts_corr",
    "ts_decay_linear", "ts_delta", "ts_delay", "ts_mean", "ts_product",
    "ts_quantile", "ts_rank", "ts_regression", "ts_scale", "ts_std_dev",
    "ts_sum", "ts_zscore", "ts_count_nans", "ts_covariance", "ts_step",
})


def _extract_operator_skeleton(expression: str) -> str:
    """Extract the operator skeleton from an expression.

    Replaces all field names and numeric literals with placeholders,
    preserving the operator structure.

    Example:
        group_rank(ts_rank(equity/cap, 60), subindustry)
        -> group_rank(ts_rank(FIELD, NUM), GROUP)

        rank(ts_mean(close, 10) - close)
        -> rank(ts_mean(FIELD, NUM) - FIELD)
    """
    # Tokenize: split on parentheses, commas, and whitespace while
    # keeping delimiters. Then classify each token.
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_/]*|\d+\.?\d*|[(),\-+/]', expression)

    result = []
    for tok in tokens:
        if tok in ("(", ")", ","):
            result.append(tok)
        elif tok in ("-", "+", "/"):
            result.append(tok)
        elif re.match(r'\d+\.?\d*$', tok):
            result.append("NUM")
        elif tok in VALID_OPERATORS:
            # Check if it's a group token (subindustry, industry, etc.)
            if tok in ("subindustry", "industry", "sector", "market", "none"):
                result.append("GROUP")
            elif tok in ("cap", "close", "open", "high", "low", "volume",
                         "vwap", "returns", "adv20"):
                result.append("FIELD")
            else:
                result.append(tok)
        else:
            # Unknown identifier — treat as a field name
            result.append("FIELD")

    # Rebuild expression string with proper spacing
    skeleton = ""
    prev = ""
    for tok in result:
        if tok in ("(", ")"):
            skeleton += tok
        elif tok == ",":
            skeleton += tok + " "
        elif tok in ("-", "+", "/"):
            skeleton += " " + tok + " "
        elif prev == "(" or prev == ",":
            skeleton += tok
        else:
            skeleton += tok
        prev = tok

    return skeleton


def _extract_operators(expression: str) -> List[str]:
    """Extract all operator names from an expression, in order of appearance.

    Returns only tokens that are in VALID_OPERATORS and are actual operators
    (not group/field tokens).
    """
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_/]*', expression)
    operators = []
    for tok in tokens:
        if tok in VALID_OPERATORS and tok not in (
            "subindustry", "industry", "sector", "market", "none",
            "cap", "close", "open", "high", "low", "volume", "vwap",
            "returns", "adv20",
        ):
            operators.append(tok)
    return operators


def _extract_fields(expression: str) -> List[str]:
    """Extract all field names from an expression.

    Fields are identifiers that are NOT operators, NOT group tokens,
    and NOT numeric literals.
    """
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_/]*', expression)
    fields = []
    for tok in tokens:
        if tok not in VALID_OPERATORS and not re.match(r'\d+\.?\d*$', tok):
            fields.append(tok)
    return fields


class MyAlphaPool:
    """Manages the pool of existing submitted alphas for dedup.

    Fetches all submitted alphas from WQ BRAIN via paginated API,
    caches them locally in JSON, and provides structural duplicate
    detection.

    Args:
        session: An authenticated ``SingleSession`` instance.
        cache_path: Path to the local JSON cache file.
            Defaults to ``my_alphas_cache.json``.
    """

    def __init__(
        self,
        session: Any,
        cache_path: str = "my_alphas_cache.json",
        submitted_cache_path: str = "submitted_alphas.json",
    ) -> None:
        self._session = session
        self.cache_path = cache_path
        self.submitted_cache_path = submitted_cache_path
        self._alphas: List[Dict[str, Any]] = []
        self._submitted: List[Dict[str, Any]] = []  # final submitted only (for correlation)
        self._signatures: List[Dict[str, str]] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, force_refresh: bool = False) -> None:
        """Load alphas from cache or fetch from API.

        Also loads final submitted alphas from ``submitted_alphas.json``
        for complete dedup coverage.

        Args:
            force_refresh: If True, always fetch from API even if cache
                exists.
        """
        if force_refresh or not os.path.exists(self.cache_path):
            self._fetch_all()
            self._save_cache()
        else:
            self._load_cache()

        # Also load final submitted alphas for dedup
        self._load_submitted_alphas()

        self._build_signatures()
        self._loaded = True

    def is_duplicate(
        self, expression: str, settings: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """Check if an expression duplicates an existing submitted alpha.

        Uses 3-level classification:

        1. ``exact_duplicate`` — identical expression string (ignoring
           whitespace differences).
        2. ``structural_duplicate`` — same operator skeleton but different
           field names (e.g. ``group_rank(ts_rank(equity/cap, 60), subindustry)``
           vs ``group_rank(ts_rank(ebit/cap, 60), subindustry)``).
        3. ``new_alpha`` — different operator family and different core
           fields.

        Args:
            expression: The alpha expression to check.
            settings: Optional settings dict (not used for dedup currently,
                but reserved for future settings-level comparison).

        Returns:
            Tuple of (is_duplicate: bool, reason: str).
            ``reason`` is one of ``"exact_duplicate"``,
            ``"structural_duplicate"``, or ``"new_alpha"``.
        """
        if not self._loaded:
            self.load()

        # Normalize whitespace for comparison: collapse multiple spaces,
        # then remove spaces around parentheses (but keep comma-space).
        # This ensures "group_rank( ts_rank( x , 60 ) , subindustry )"
        # matches "group_rank(ts_rank(x, 60), subindustry)".
        normalized = " ".join(expression.split())
        # Remove spaces before/after parentheses
        normalized = re.sub(r'\s*\(\s*', '(', normalized)
        normalized = re.sub(r'\s*\)\s*', ')', normalized)

        # Level 1: Exact expression match
        for alpha in self._alphas:
            existing_expr = alpha.get("regular", "")
            if not existing_expr:
                continue
            normalized_existing = " ".join(existing_expr.split())
            if normalized == normalized_existing:
                return (True, "exact_duplicate")

        # Level 2: Structural pattern match (same operator skeleton)
        candidate_skeleton = _extract_operator_skeleton(expression)
        for alpha in self._alphas:
            existing_expr = alpha.get("regular", "")
            if not existing_expr:
                continue
            existing_skeleton = _extract_operator_skeleton(existing_expr)
            if candidate_skeleton == existing_skeleton:
                return (True, "structural_duplicate")

        # Level 3: Different family + different core fields → new alpha
        return (False, "new_alpha")

    def suggest_pivot(
        self, expression: str, settings: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Suggest pivot variants for a structurally duplicate expression.

        For expressions that are structural duplicates of existing alphas,
        generate variants that change the operator family to reduce
        correlation. Returns up to 3 pivot suggestions.

        Pivot strategies (in priority order):
        1. **transformation_swap** — Replace top-level operator with a
           correlation-reducing alternative:
           - rank ↔ zscore ↔ group_zscore
           - ts_rank ↔ ts_zscore
        2. **grouping_change** — Change group parameter:
           - subindustry → industry → market → none
        3. **time_horizon_change** — Double or halve numeric lookback
           parameters.

        Args:
            expression: The alpha expression to pivot.
            settings: Optional settings dict (used for grouping info).

        Returns:
            List of pivoted expression strings (max 3).
        """
        if not self._loaded:
            self.load()

        pivots: List[str] = []

        # Strategy 1: transformation_swap
        swap_map = {
            "rank": ["zscore", "group_zscore"],
            "zscore": ["rank", "group_zscore"],
            "group_rank": ["group_zscore"],
            "group_zscore": ["group_rank"],
            "ts_rank": ["ts_zscore"],
            "ts_zscore": ["ts_rank"],
        }

        # Find the outermost operator to swap
        expr_stripped = expression.strip()
        for op, replacements in swap_map.items():
            # Match operator at the start of expression followed by (
            pattern = re.compile(r'^' + re.escape(op) + r'\(')
            if pattern.match(expr_stripped):
                for replacement in replacements:
                    pivoted = pattern.sub(replacement + '(', expr_stripped, count=1)
                    if pivoted != expression and pivoted not in pivots:
                        pivots.append(pivoted)
                break  # Only swap the outermost operator

        # Strategy 2: grouping_change
        group_map = {
            "subindustry": ["industry", "market"],
            "industry": ["subindustry", "market"],
            "sector": ["subindustry", "industry"],
            "market": ["subindustry", "industry"],
        }

        for group_from, group_tos in group_map.items():
            # Match group parameter at end of expression: , group_from)
            # or , group_from, ...)
            pattern = re.compile(
                r',\s*' + re.escape(group_from) + r'\s*\)'
            )
            match = pattern.search(expression)
            if match:
                for group_to in group_tos:
                    pivoted = expression[:match.start()] + ", " + group_to + ")"
                    if pivoted not in pivots:
                        pivots.append(pivoted)
                break

        # Strategy 3: time_horizon_change — double or halve numeric lookbacks
        # Find numeric parameters after known TS operators
        horizon_map = {
            "5": ["10"],
            "10": ["5", "20"],
            "20": ["10", "40"],
            "30": ["15", "60"],
            "40": ["20", "60"],
            "60": ["30", "120"],
        }

        # Find all (operator, number) pairs in the expression
        ts_pattern = re.compile(
            r'(ts_mean|ts_rank|ts_delta|ts_std_dev|ts_decay_linear|ts_sum|'
            r'ts_zscore|ts_corr|ts_product|ts_arg_max|ts_arg_min|ts_av_diff|'
            r'ts_backfill|ts_quantile|ts_scale)\s*\(\s*([^,]+)\s*,\s*(\d+)\s*'
        )
        for m in ts_pattern.finditer(expression):
            num = m.group(3)
            if num in horizon_map:
                for replacement_num in horizon_map[num]:
                    start = m.start(3)
                    end = m.end(3)
                    pivoted = expression[:start] + replacement_num + expression[end:]
                    if pivoted not in pivots:
                        pivots.append(pivoted)

        # Return max 3 pivots
        return pivots[:3]

    def get_expression_signatures(self) -> List[Dict[str, str]]:
        """Return structural signatures of all existing alphas.

        Each signature contains:
        - ``alpha_id``: The WQ alpha ID.
        - ``expression``: The original expression string.
        - ``skeleton``: The operator skeleton (e.g.,
          ``group_rank(ts_rank(FIELD, NUM), GROUP)``).
        - ``operators``: List of operators used, in order.

        Returns:
            List of signature dicts.
        """
        if not self._loaded:
            self.load()
        return list(self._signatures)

    @property
    def alphas(self) -> List[Dict[str, Any]]:
        """Return the raw list of fetched alpha dicts."""
        if not self._loaded:
            self.load()
        return list(self._alphas)

    @property
    def submitted(self) -> List[Dict[str, Any]]:
        """Return only the final submitted alphas (for correlation reference).

        These are loaded from ``submitted_alphas.json`` and represent
        the real final submissions, not test attempts.
        """
        if not self._loaded:
            self.load()
        return list(self._submitted)

    @property
    def count(self) -> int:
        """Number of alphas in the pool."""
        if not self._loaded:
            self.load()
        return len(self._alphas)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_all(self) -> None:
        """Fetch all submitted alphas from WQ BRAIN API with pagination.

        Uses GET /alphas?order=-created&limit=50&offset={x} and loops
        offset by 50 until no more results are returned.

        Handles 0 submitted alphas gracefully (valid initial state).
        """
        self._alphas = []
        offset = 0
        limit = 50

        while True:
            url = (
                f"{ace_lib.brain_api_url}"
                f"/users/self/alphas?limit={limit}&offset={offset}"
            )

            for attempt in range(5):
                try:
                    response = self._session.get(url)

                    # Handle rate limiting (429)
                    if response.status_code == 429:
                        retry_after = float(response.headers.get("Retry-After", "60"))
                        logger.warning(
                            "Rate limited (429) fetching alphas at offset %d — "
                            "retrying after %.0fs (attempt %d/5)",
                            offset, retry_after, attempt + 1,
                        )
                        import time
                        time.sleep(retry_after)
                        continue

                    response.raise_for_status()
                    break
                except Exception as exc:
                    if attempt < 4:
                        logger.warning(
                            "Error fetching alphas at offset %d (attempt %d/5): %s",
                            offset, attempt + 1, exc,
                        )
                        import time
                        time.sleep(5)
                    else:
                        logger.error(
                            "Failed to fetch alphas at offset %d after 5 attempts: %s",
                            offset, exc,
                        )
                        raise
            else:
                # All retries exhausted
                logger.error("All retries exhausted for offset %d", offset)
                break

            data = response.json()

            # Handle empty results (0 submitted alphas)
            if not data or "results" not in data:
                logger.info("No alphas found at offset %d", offset)
                break

            results = data.get("results", [])
            if not results:
                break

            for alpha_data in results:
                # Extract the expression and settings
                regular = ""
                settings = {}

                # The expression is in alpha_data["regular"]["code"] for
                # regular alphas, or alpha_data["regular"] as a string
                if isinstance(alpha_data.get("regular"), dict):
                    regular = alpha_data["regular"].get("code", "")
                elif isinstance(alpha_data.get("regular"), str):
                    regular = alpha_data["regular"]

                # Settings are in alpha_data["settings"]
                if isinstance(alpha_data.get("settings"), dict):
                    settings = alpha_data["settings"]

                self._alphas.append({
                    "id": alpha_data.get("id", ""),
                    "regular": regular,
                    "settings": settings,
                    "date_created": alpha_data.get("date_created", ""),
                    "status": alpha_data.get("status", ""),
                })

            # Check if there are more pages
            count = data.get("count", 0)
            offset += limit
            if offset >= count:
                break

        logger.info("Fetched %d alphas from WQ BRAIN", len(self._alphas))

    def _load_cache(self) -> None:
        """Load alphas from local JSON cache file."""
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                self._alphas = data
            elif isinstance(data, dict) and "alphas" in data:
                self._alphas = data["alphas"]
            else:
                logger.warning(
                    "Unexpected cache format in %s — starting fresh",
                    self.cache_path,
                )
                self._alphas = []
        except json.JSONDecodeError as exc:
            logger.warning(
                "Corrupt cache file at %s (%s) — starting fresh",
                self.cache_path, exc,
            )
            self._alphas = []
        except OSError as exc:
            logger.warning(
                "Cannot read cache file at %s (%s) — starting fresh",
                self.cache_path, exc,
            )
            self._alphas = []

    def _load_submitted_alphas(self) -> None:
        """Load final submitted alphas from submitted_alphas.json.

        Merges them into self._alphas for complete dedup coverage.
        Also stores them separately in self._submitted for correlation reference.
        Only alphas with status in VALID_SUBMITTED_STATUSES are included in
        self._submitted — alphas still in draft/simulated/check-passed state
        are excluded because they have not been fully submitted to WQ.
        """
        self._submitted = []
        if not os.path.exists(self.submitted_cache_path):
            logger.info(
                "No submitted alphas cache at %s — skipping",
                self.submitted_cache_path,
            )
            return

        try:
            with open(self.submitted_cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            submitted = data.get("alphas", [])
            if not submitted:
                return

            # Store only truly submitted (ACTIVE) alphas for correlation reference.
            # Alphas with status "draft", "simulated", "check-passed", etc.
            # have NOT been submitted to WQ and must not be treated as submitted.
            self._submitted = [
                a for a in submitted
                if a.get("status", "").lower() in VALID_SUBMITTED_STATUSES
            ]

            # Merge into main pool for dedup, avoiding duplicate IDs
            existing_ids = {a.get("id") for a in self._alphas if a.get("id")}
            count_added = 0
            for a in submitted:
                aid = a.get("id")
                if aid and aid not in existing_ids:
                    self._alphas.append(a)
                    existing_ids.add(aid)
                    count_added += 1

            n_filtered = len(self._submitted)
            logger.info(
                "Loaded %d alphas from %s (%d submitted/active, %d new, %d skipped)",
                len(submitted), self.submitted_cache_path,
                n_filtered, count_added, len(submitted) - count_added,
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Cannot load submitted alphas from %s (%s)",
                self.submitted_cache_path, exc,
            )
            self._submitted = []

    def _save_cache(self) -> None:
        """Write alphas to local JSON cache file."""
        cache_data = {
            "version": 1,
            "fetched_at": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "count": len(self._alphas),
            "alphas": self._alphas,
        }
        tmp_path = self.cache_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(cache_data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.cache_path)
        except OSError as exc:
            logger.error("Failed to save cache to %s: %s", self.cache_path, exc)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def _build_signatures(self) -> None:
        """Build structural signatures for all alphas in the pool."""
        self._signatures = []
        for alpha in self._alphas:
            expr = alpha.get("regular", "")
            if not expr:
                continue
            skeleton = _extract_operator_skeleton(expr)
            operators = _extract_operators(expr)
            self._signatures.append({
                "alpha_id": alpha.get("id", ""),
                "expression": expr,
                "skeleton": skeleton,
                "operators": operators,
            })