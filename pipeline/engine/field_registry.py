# ⛔ READ-ONLY — DO NOT MODIFY. Protected engine module.
"""
field_registry.py — Platform datafield caching, validation, and pagination.

FieldRegistry handles:
  - Fetching all datafields from WQ BRAIN API with pagination
  - Caching to fields_cache.json for aggressive caching
  - Expression validation (identifying unknown field/operator tokens)
  - Field search and category-based alternative lookup
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from . import ace_lib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VALID_OPERATORS — complete set from operators.md
# ---------------------------------------------------------------------------
VALID_OPERATORS: Set[str] = {
    # Arithmetic
    "abs", "add", "densify", "divide", "inverse", "log", "max", "min",
    "multiply", "power", "reverse", "sign", "signed_power", "sqrt", "subtract",
    # Logical
    "and", "if_else", "is_nan", "not", "or",
    # Time Series
    "days_from_last_change", "hump", "kth_element", "last_diff_value",
    "ts_arg_max", "ts_arg_min", "ts_av_diff", "ts_backfill", "ts_corr",
    "ts_count_nans", "ts_covariance", "ts_decay_linear", "ts_delay", "ts_delta",
    "ts_mean", "ts_product", "ts_quantile", "ts_rank", "ts_regression",
    "ts_scale", "ts_std_dev", "ts_step", "ts_sum", "ts_zscore",
    # Cross Sectional
    "normalize", "quantile", "rank", "scale", "winsorize", "zscore",
    # Vector
    "vec_avg", "vec_sum",
    # Transformational
    "bucket", "trade_when",
    # Group
    "group_backfill", "group_mean", "group_neutralize", "group_rank",
    "group_scale", "group_zscore",
}

# Common parameters / field aliases that appear in expressions but are not
# datafield IDs — they are grouping constants or built-in price fields.
COMMON_PARAMS: Set[str] = {
    "subindustry", "industry", "sector", "market", "none",
    "cap", "close", "open", "high", "low", "volume", "vwap", "returns", "adv20",
}

# Numeric-like pattern (matches integers and floats, including negatives)
_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


class FieldRegistry:
    """Caches WQ BRAIN datafields and validates alpha expressions.

    Usage:
        session = ace_lib.start_session()
        fr = FieldRegistry(session)
        is_valid, unknowns = fr.validate_expression("rank(ts_mean(close, 10) - close)")
    """

    def __init__(
        self,
        session: Optional[ace_lib.SingleSession],
        *,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        cache_path: str = "fields_cache.json",
    ):
        self.session = session
        self.region = region
        self.universe = universe
        self.delay = delay
        self.cache_path = Path(cache_path)

        # Internal field storage: dict mapping field_id -> field metadata dict
        self._fields: Dict[str, dict] = {}
        # Category index: dict mapping category -> set of field_ids
        self._category_index: Dict[str, Set[str]] = {}

        self._load_or_fetch()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def fields(self) -> Dict[str, dict]:
        """Return the full field dictionary {field_id: metadata_dict}."""
        return self._fields

    @property
    def field_ids(self) -> Set[str]:
        """Return the set of all known field IDs."""
        return set(self._fields.keys())

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _load_cache(self) -> bool:
        """Load fields from cache file. Returns True if cache was loaded."""
        if not self.cache_path.exists():
            logger.info("No fields cache found at %s", self.cache_path)
            return False

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._fields = data.get("fields", {})
            self._rebuild_category_index()
            logger.info("Loaded %d fields from cache %s", len(self._fields), self.cache_path)
            return True
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Corrupt fields cache at %s: %s — will re-fetch", self.cache_path, exc)
            return False

    def _save_cache(self) -> None:
        """Persist current fields dict to cache file."""
        payload = {
            "version": 1,
            "region": self.region,
            "universe": self.universe,
            "delay": self.delay,
            "count": len(self._fields),
            "fields": self._fields,
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d fields to cache %s", len(self._fields), self.cache_path)

    # ------------------------------------------------------------------
    # API fetch with pagination
    # ------------------------------------------------------------------

    def _fetch_all(self) -> int:
        """Fetch ALL datafields from the API using pagination.

        Handles BOTH modes:
          - search="" (empty): first response contains ``count``, paginate up to that count
          - search="keyword": count defaults to 100, paginate with search param

        We fetch ALL types (MATRIX + VECTOR) because some VECTOR fields are useful.

        Returns:
            Number of fields fetched.
        """
        if self.session is None:
            logger.warning("No session provided — cannot fetch fields from API")
            return 0

        # Fetch MATRIX type fields
        matrix_fields = self._fetch_by_type(data_type="MATRIX")
        # Fetch VECTOR type fields (some are useful)
        vector_fields = self._fetch_by_type(data_type="VECTOR")

        # Merge into existing fields (CSV-loaded fields act as base)
        fetched = 0
        for field in vector_fields:
            fid = field.get("id")
            if fid and fid not in self._fields:
                self._fields[fid] = self._normalize_field(field)
                fetched += 1
        for field in matrix_fields:
            fid = field.get("id")
            if fid and fid not in self._fields:
                self._fields[fid] = self._normalize_field(field)
                fetched += 1

        self._rebuild_category_index()
        self._save_cache()
        logger.info("Fetched %d new fields from API (total %d)", fetched, len(self._fields))
        return fetched

    def _fetch_by_type(self, data_type: str = "MATRIX", search: str = "") -> List[dict]:
        """Fetch fields of a specific data_type with pagination.

        Mirrors ace_lib.get_datafields() pagination logic but returns raw
        list of dicts instead of DataFrame, and handles both URL templates.
        """
        type_param = f"&type={data_type}" if data_type != "ALL" else ""
        base_url = (
            ace_lib.brain_api_url
            + "/data-fields?"
            + "&instrumentType=EQUITY"
            + f"&region={self.region}&delay={self.delay}&universe={self.universe}{type_param}"
        )

        if len(search) == 0:
            # No-search mode: URL includes limit=50 and offset={x}
            # First request to get total count
            url_first = base_url + "&dataset.id=&limit=50&offset=0"
            max_try = 5
            resp = None
            for attempt in range(max_try):
                try:
                    resp = self.session.get(url_first)
                    if resp.status_code == 200 and "results" in resp.json():
                        break
                except Exception as exc:
                    logger.warning("Fetch attempt %d failed: %s", attempt + 1, exc)
                time.sleep(5)
            else:
                logger.error("Failed to fetch first page of datafields after %d attempts", max_try)
                return []

            data = resp.json()
            count = data.get("count", 0)
            if count == 0:
                logger.warning("No fields found for type=%s", data_type)
                return []

            url_template = base_url + "&dataset.id=&limit=50" + "&offset={x}"
        else:
            # Search mode: count defaults to 100
            count = 100
            url_template = (
                base_url
                + f"&limit=50&search={search}"
                + "&offset={x}"
            )

        # Paginate through all results
        all_fields: List[dict] = []
        for offset in range(0, count, 50):
            url = url_template.format(x=offset)
            max_try = 5
            for attempt in range(max_try):
                try:
                    resp = self.session.get(url)
                    if resp.status_code == 200 and "results" in resp.json():
                        break
                except Exception as exc:
                    logger.warning("Pagination fetch attempt %d at offset %d failed: %s",
                                   attempt + 1, offset, exc)
                time.sleep(5)
            else:
                logger.error("Failed to fetch offset %d after %d attempts", offset, max_try)
                continue

            results = resp.json().get("results", [])
            if not results:
                # No more results — we've exhausted the pages
                break
            all_fields.extend(results)

        return all_fields

    # ------------------------------------------------------------------
    # Load-or-fetch
    # ------------------------------------------------------------------

    def _load_or_fetch(self) -> None:
        """Load from local CSV fields first, then supplement with API fetch.

        1. Loads field→category mappings from all CSVs under ``Data and operators/``
           (regardless of region/universe/delay settings — gives access to ALL fields).
        2. Attempts API fetch with the configured region/universe/delay for richer metadata.
        """
        # Always load local CSV fields first (all 8k+ fields regardless of API filters)
        self._load_csv_fields()

        # Supplement with API data if available
        if self.session is not None:
            logger.info("Fetching API fields (region=%s, universe=%s, delay=%d)...",
                        self.region, self.universe, self.delay)
            api_fetched = self._fetch_all()
            if api_fetched:
                logger.info("Merged %d API fields with CSV fields", api_fetched)
                self._save_cache()
        else:
            logger.info("No session — using CSV fields only (%d fields)", len(self._fields))
    def _load_csv_fields(self) -> int:
        """Load field IDs from local CSVs under ``Data and operators/``.

        Each field is stored with minimal metadata (id, category) so the
        registry has access to ALL fields regardless of API region/universe/delay
        filters.  Returns the number of fields loaded.
        """
        import csv
        csv_dir = Path(__file__).parent.parent / "Data and operators"
        if not csv_dir.exists():
            logger.warning("CSV directory not found: %s", csv_dir)
            return 0

        count = 0
        for csv_path in sorted(csv_dir.glob("*.csv")):
            try:
                with open(csv_path, encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        fid = row.get("id", "").strip()
                        if not fid:
                            continue
                        cat_str = row.get("category", "uncategorized")
                        # Extract simple category name from JSON-like strings
                        cat = "uncategorized"
                        if cat_str:
                            try:
                                parsed = json.loads(cat_str.replace("'", '"'))
                                cat = parsed.get("name", "uncategorized")
                            except (json.JSONDecodeError, AttributeError):
                                cat = str(cat_str).strip()
                        self._fields[fid] = {"id": fid, "category": cat, "source": "csv"}
                        count += 1
            except Exception as exc:
                logger.warning("Failed to read %s: %s", csv_path, exc)

        self._rebuild_category_index()
        logger.info("Loaded %d fields from %d local CSVs", count, len(list(csv_dir.glob("*.csv"))))
        return count
    # Category index
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_field(field: dict) -> dict:
        """Normalize an API-fetched field dict to a stable inner format.
        WQ API returns ``category`` as ``{"name": "fundamental", "id": ...}``
        which is unhashable.  Extract just the name string."""
        cat_raw = field.get("category", "uncategorized")
        if isinstance(cat_raw, dict):
            cat = cat_raw.get("name", "uncategorized")
        else:
            cat = str(cat_raw).strip() if cat_raw else "uncategorized"
        return {"id": field.get("id", ""), "category": cat, "source": "api"}

    def _rebuild_category_index(self) -> None:
        """Rebuild the category -> set(field_ids) index from current fields."""
        self._category_index = {}
        for fid, meta in self._fields.items():
            cat = meta.get("category", "uncategorized")
            if cat not in self._category_index:
                self._category_index[cat] = set()
            self._category_index[cat].add(fid)

    # ------------------------------------------------------------------
    # Expression validation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tokens(expression: str) -> List[str]:
        """Extract identifier tokens from an alpha expression.

        WQ BRAIN field names can contain ``/`` (e.g. ``equity/cap``).
        The ``/`` character is ambiguous — it is both a division operator
        and part of field identifiers.  We handle this by:

          1. Replacing known arithmetic operators (``+``, ``-``, ``*``) with
             spaces, but keeping ``/`` intact.
          2. Splitting on whitespace and punctuation (except ``/``).
          3. For each resulting token that contains ``/``, checking whether
             it looks like a compound field name (``word/word``) or an
             arithmetic division (``close / open`` with spaces already
             removed).  Because we strip spaces first, ``close/open`` is
             ambiguous — we keep it as a single token and let validation
             decide if it's a known field.
        """
        # Remove numeric literals entirely (integers, floats, negatives)
        cleaned = re.sub(r"-?\d+(\.\d+)?", " ", expression)

        # Replace punctuation and operator symbols with spaces — but NOT "/"
        # because "/" can be part of field names like "equity/cap".
        for ch in "(),+-*<>!=:":
            cleaned = cleaned.replace(ch, " ")

        # Split on whitespace
        raw_tokens = cleaned.split()

        # Keep only identifier-like tokens (start with letter/underscore,
        # may contain letters, digits, underscores, and "/")
        tokens = [
            tok for tok in raw_tokens
            if re.match(r"^[A-Za-z_][A-Za-z0-9_/]*$", tok)
        ]

        return tokens

    def validate_expression(self, expression: str) -> Tuple[bool, List[str]]:
        """Validate an alpha expression against known fields and operators.

        Returns:
            (is_valid, unknown_tokens) where:
              - is_valid is True if ALL tokens are known
              - unknown_tokens lists any tokens not in VALID_OPERATORS, COMMON_PARAMS,
                or the fetched field registry
        """
        tokens = self._extract_tokens(expression)

        # Build the full whitelist: operators + common params + known field IDs
        known = VALID_OPERATORS | COMMON_PARAMS | self.field_ids

        unknown = [tok for tok in tokens if tok not in known]

        return (len(unknown) == 0, unknown)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, keyword: str) -> List[dict]:
        """Search local field cache by keyword (case-insensitive).

        Matches against field id, description, category, and subcategory.
        """
        keyword_lower = keyword.lower()
        results = []
        for fid, meta in self._fields.items():
            searchable = " ".join([
                fid,
                meta.get("description", ""),
                meta.get("category", ""),
                meta.get("subcategory", ""),
            ]).lower()
            if keyword_lower in searchable:
                results.append({"id": fid, **meta})
        return results

    # ------------------------------------------------------------------
    # Alternatives (category-based)
    # ------------------------------------------------------------------

    def get_alternatives(self, field_id: str, same_category: bool = True) -> List[str]:
        """Find alternative fields in the same category as ``field_id``.

        Used by pivot strategy 5 (field_substitution) to find replacement
        fields that are semantically related.

        Args:
            field_id: The field to find alternatives for (e.g. "equity/cap")
            same_category: If True, only return fields in the same category

        Returns:
            List of field IDs that are alternatives, excluding the original.
        """
        if field_id not in self._fields:
            logger.warning("Field '%s' not found in registry — no alternatives", field_id)
            return []

        meta = self._fields[field_id]
        category = meta.get("category", "uncategorized")

        if not same_category or category not in self._category_index:
            # Fallback: return all other fields (limited to 20)
            return [fid for fid in list(self._fields.keys())[:20] if fid != field_id]

        # Return fields in the same category, excluding the original
        alternatives = sorted(
            fid for fid in self._category_index[category] if fid != field_id
        )
        return alternatives

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Force re-fetch from API, bypassing cache."""
        logger.info("Refreshing field registry from API (cache bypass)")
        self._fetch_all()


# ---------------------------------------------------------------------------
# Standalone test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Quick smoke test without API session
    fr = FieldRegistry(None, cache_path="fields_cache.json")

    # If cache exists, run validation tests
    if fr.fields:
        print(f"Loaded {len(fr.fields)} fields from cache")

        # Test validate_expression
        tests = [
            ("rank(ts_mean(close, 10) - close)", True, []),
            ("rank(ts_mean(sale, 10) - close)", False, ["sale"]),
            ("group_rank(ts_rank(equity/cap, 60), subindustry)", True, []),
        ]
        for expr, expected_valid, _ in tests:
            is_valid, unknowns = fr.validate_expression(expr)
            status = "PASS" if is_valid == expected_valid else "FAIL"
            print(f"  [{status}] validate({expr!r}) = ({is_valid}, {unknowns})")

        # Test search
        results = fr.search("cap")
        print(f"  search('cap') returned {len(results)} results")

        # Test get_alternatives
        if "equity/cap" in fr.fields:
            alts = fr.get_alternatives("equity/cap")
            print(f"  get_alternatives('equity/cap') = {alts[:5]}")
    else:
        print("No cached fields — run with a session to populate cache")
        print("Valid operators count:", len(VALID_OPERATORS))
        print("Common params count:", len(COMMON_PARAMS))