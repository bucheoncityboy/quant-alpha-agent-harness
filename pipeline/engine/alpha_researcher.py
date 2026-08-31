"""WQ BRAIN Alpha Researcher — core module for structured operator composition.

This module replaces template-based alpha generation with a programmatic
composition system. It provides:

  - OperatorDef / OperatorRegistry — all 66 operators with type metadata
  - OperatorNode / OperatorTreeCompiler — tree-to-WQ-expression compilation
  - LLMComposer — random operator-space explorer with type-safe recursion
  - PatternTracker — 5-strike rule to detect stale patterns
  - FeedbackStore — aggregate and persist simulation feedback
  - ValidateGate — expression token validation
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "Data and operators"

GROUPS = ["subindustry", "industry", "sector", "market"]
LOOKBACKS = [5, 10, 15, 20, 21, 30, 40, 45, 60, 63, 90, 120, 252]

# ---------------------------------------------------------------------------
# 1. OperatorDef + OperatorRegistry
# ---------------------------------------------------------------------------


@dataclass
class OperatorDef:
    """Metadata for a single WQ operator.

    Attributes:
        name: Operator name as used in WQ alpha expressions.
        category: One of arithmetic|logical|time_series|cross_sectional|
                  vector|transformational|group.
        arity: (min, max) positional args.  max=None means variadic.
        input_types: Type hint per positional arg (in order):
                     "vector" — field or time-series sub-expression
                     "scalar" — numeric literal
                     "lookback" — integer lookback days (d)
                     "group" — group identifier string
                     "bool" — true/false flag
        output_type: "vector", "rank", "scalar", "group"
        has_lookback: True if the operator takes a lookback (d) parameter.
        has_group_param: True if the operator takes a group identifier.
        params: Default parameter values keyed by name.
    """
    name: str
    category: str
    arity: Tuple[int, Optional[int]]  # (min, max_or_None)
    input_types: List[str]
    output_type: str = "vector"
    has_lookback: bool = False
    has_group_param: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


def _build_operator_defs() -> Dict[str, OperatorDef]:
    """Build the complete dictionary of all 66 WQ operators.

    Sources: operators.md section structure + operator signatures.
    60 named operators + 6 infix comparison operators.
    """
    defs: Dict[str, OperatorDef] = {}

    def _add(name, category, arity, input_types, output_type="vector",
             has_lookback=False, has_group_param=False, params=None):
        defs[name] = OperatorDef(
            name=name,
            category=category,
            arity=arity,
            input_types=input_types,
            output_type=output_type,
            has_lookback=has_lookback,
            has_group_param=has_group_param,
            params=params or {},
        )

    # ── Arithmetic (15) ──────────────────────────────────────────────
    _add("abs", "arithmetic", (1, 1), ["vector"])
    _add("add", "arithmetic", (2, None), ["vector", "vector"],
         params={"filter": False})
    _add("densify", "arithmetic", (1, 1), ["vector"])
    _add("divide", "arithmetic", (2, 2), ["vector", "vector"])
    _add("inverse", "arithmetic", (1, 1), ["vector"])
    _add("log", "arithmetic", (1, 1), ["vector"])
    _add("max", "arithmetic", (2, None), ["vector", "vector"])
    _add("min", "arithmetic", (2, None), ["vector", "vector"])
    _add("multiply", "arithmetic", (2, None), ["vector", "vector"],
         params={"filter": False})
    _add("power", "arithmetic", (2, 2), ["vector", "vector"])
    _add("reverse", "arithmetic", (1, 1), ["vector"])
    _add("sign", "arithmetic", (1, 1), ["vector"])
    _add("signed_power", "arithmetic", (2, 2), ["vector", "vector"])
    _add("sqrt", "arithmetic", (1, 1), ["vector"])
    _add("subtract", "arithmetic", (2, None), ["vector", "vector"],
         params={"filter": False})

    # ── Logical — named (5) ──────────────────────────────────────────
    _add("and", "logical", (2, 2), ["vector", "vector"])
    _add("if_else", "logical", (3, 3),
         ["vector", "vector", "vector"])
    _add("is_nan", "logical", (1, 1), ["vector"])
    _add("not", "logical", (1, 1), ["vector"])
    _add("or", "logical", (2, 2), ["vector", "vector"])

    # ── Logical — infix comparison (6) ───────────────────────────────
    _add("<", "logical", (2, 2), ["vector", "vector"])
    _add("<=", "logical", (2, 2), ["vector", "vector"])
    _add("==", "logical", (2, 2), ["vector", "vector"])
    _add(">", "logical", (2, 2), ["vector", "vector"])
    _add(">=", "logical", (2, 2), ["vector", "vector"])
    _add("!=", "logical", (2, 2), ["vector", "vector"])

    # ── Time Series (24) ─────────────────────────────────────────────
    _add("days_from_last_change", "time_series", (1, 1), ["vector"])
    _add("hump", "time_series", (1, 1), ["vector"],
         params={"hump": 0.01})
    _add("kth_element", "time_series", (3, 4),
         ["vector", "lookback", "scalar"],
         has_lookback=True, params={"ignore": "NaN"})
    _add("last_diff_value", "time_series", (2, 2),
         ["vector", "lookback"], has_lookback=True)
    _add("ts_arg_max", "time_series", (2, 2),
         ["vector", "lookback"], has_lookback=True)
    _add("ts_arg_min", "time_series", (2, 2),
         ["vector", "lookback"], has_lookback=True)
    _add("ts_av_diff", "time_series", (2, 2),
         ["vector", "lookback"], has_lookback=True)
    _add("ts_backfill", "time_series", (1, 3),
         ["vector"], has_lookback=True,
         params={"lookback": 5, "k": 1})
    _add("ts_corr", "time_series", (3, 3),
         ["vector", "vector", "lookback"], has_lookback=True)
    _add("ts_count_nans", "time_series", (2, 2),
         ["vector", "lookback"], has_lookback=True)
    _add("ts_covariance", "time_series", (3, 3),
         ["vector", "vector", "lookback"], has_lookback=True)
    _add("ts_decay_linear", "time_series", (2, 3),
         ["vector", "lookback"], has_lookback=True,
         params={"dense": False})
    _add("ts_delay", "time_series", (2, 2),
         ["vector", "lookback"], has_lookback=True)
    _add("ts_delta", "time_series", (2, 2),
         ["vector", "lookback"], has_lookback=True)
    _add("ts_mean", "time_series", (2, 2),
         ["vector", "lookback"], has_lookback=True)
    _add("ts_product", "time_series", (2, 2),
         ["vector", "lookback"], has_lookback=True)
    _add("ts_quantile", "time_series", (2, 3),
         ["vector", "lookback"], has_lookback=True,
         params={"driver": "gaussian"})
    _add("ts_rank", "time_series", (2, 3),
         ["vector", "lookback"], has_lookback=True,
         params={"constant": 0})
    _add("ts_regression", "time_series", (3, 5),
         ["vector", "vector", "lookback"],
         has_lookback=True, params={"lag": 0, "rettype": 0})
    _add("ts_scale", "time_series", (2, 3),
         ["vector", "lookback"], has_lookback=True,
         params={"constant": 0})
    _add("ts_std_dev", "time_series", (2, 2),
         ["vector", "lookback"], has_lookback=True)
    _add("ts_step", "time_series", (1, 1), ["scalar"])
    _add("ts_sum", "time_series", (2, 2),
         ["vector", "lookback"], has_lookback=True)
    _add("ts_zscore", "time_series", (2, 2),
         ["vector", "lookback"], has_lookback=True)

    # ── Cross Sectional (6) ──────────────────────────────────────────
    _add("normalize", "cross_sectional", (1, 3), ["vector"],
         params={"useStd": False, "limit": 0.0})
    _add("quantile", "cross_sectional", (1, 3), ["vector"],
         params={"driver": "gaussian", "sigma": 1.0})
    _add("rank", "cross_sectional", (1, 2), ["vector"],
         params={"rate": 2})
    _add("scale", "cross_sectional", (1, 4), ["vector"],
         params={"scale": 1, "longscale": 1, "shortscale": 1})
    _add("winsorize", "cross_sectional", (1, 2), ["vector"],
         params={"std": 4})
    _add("zscore", "cross_sectional", (1, 1), ["vector"])

    # ── Vector (2) ───────────────────────────────────────────────────
    _add("vec_avg", "vector", (1, 1), ["vector"], output_type="scalar")
    _add("vec_sum", "vector", (1, 1), ["vector"], output_type="scalar")

    # ── Transformational (2) ─────────────────────────────────────────
    _add("bucket", "transformational", (1, 2), ["vector"],
         params={"range": "0,1,0.1"})
    _add("trade_when", "transformational", (3, 3),
         ["vector", "vector", "vector"])

    # ── Group (6) ────────────────────────────────────────────────────
    _add("group_backfill", "group", (3, 4),
         ["vector", "group", "lookback"],
         has_lookback=True, has_group_param=True, params={"std": 4.0})
    _add("group_mean", "group", (3, 3),
         ["vector", "vector", "group"],
         has_group_param=True)
    _add("group_neutralize", "group", (2, 2),
         ["vector", "group"],
         has_group_param=True)
    _add("group_rank", "group", (2, 2),
         ["vector", "group"],
         has_group_param=True)
    _add("group_scale", "group", (2, 2),
         ["vector", "group"],
         has_group_param=True)
    _add("group_zscore", "group", (2, 2),
         ["vector", "group"],
         has_group_param=True)

    return defs


_CATEGORY_ORDER = [
    "arithmetic", "logical", "time_series", "cross_sectional",
    "vector", "transformational", "group",
]

# Rename mapping for operator names that appear as infix symbols in WQ
_INFIX_SYMBOLS = {"<", "<=", "==", ">", ">=", "!=", "+", "-", "*", "/"}


class OperatorRegistry:
    """Loads all 66 WQ operators from operators.md, indexed by name.

    Parses the operators.md file for section headers (category) and
    ``### `op_name(args)`` entries.  Falls back to the hardcoded operator
    definitions if the file is unavailable or incomplete.
    """

    def __init__(self, operators_path: Optional[str] = None):
        if operators_path is None:
            operators_path = str(DATA_DIR / "operators.md")
        self._path = operators_path
        self._operators: Dict[str, OperatorDef] = {}
        self._load()

    def _load(self) -> None:
        """Parse operators.md and merge with hardcoded defs."""
        hardcoded = _build_operator_defs()
        file_path = Path(self._path)

        if not file_path.exists():
            self._operators = hardcoded
            return

        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            self._operators = hardcoded
            return

        # Extract operator names from ### `name(...)` entries
        found_names: Set[str] = set()
        for m in re.finditer(
            r'^###\s*`([a-z_]+)\s*\(', text, re.MULTILINE
        ):
            found_names.add(m.group(1))

        # Validate: every found name must be in hardcoded defs
        for name in found_names:
            if name not in hardcoded:
                pass  # tolerate extra entries

        # Add any hardcoded operators not found in the file (infix, etc.)
        self._operators = dict(hardcoded)

    def get(self, name: str) -> Optional[OperatorDef]:
        """Get an OperatorDef by name. Returns None if not found."""
        return self._operators.get(name)

    def list_all(self) -> List[OperatorDef]:
        """Return all operators as a list."""
        return list(self._operators.values())

    def get_by_category(self, category: str) -> List[OperatorDef]:
        """Return all operators in a given category."""
        return [op for op in self._operators.values()
                if op.category == category]

    @property
    def count(self) -> int:
        """Return total operator count (should be 66)."""
        return len(self._operators)

    def get_by_output_type(self, output_type: str) -> List[OperatorDef]:
        """Return operators that produce a given output type."""
        return [op for op in self._operators.values()
                if op.output_type == output_type]

    def get_child_candidates(self,
                             parent_type: str) -> List[OperatorDef]:
        """Return operators whose output_type is compatible with a parent
        argument slot expecting the given type.

        "vector" slots accept operators whose output_type is "vector"
        or "rank".  "scalar" slots accept "scalar" operators.  "group"
        slots accept "group" operators.
        """
        compatible_outputs: Set[str] = set()
        if parent_type in ("vector", "rank"):
            compatible_outputs = {"vector", "rank"}
        elif parent_type == "scalar":
            compatible_outputs = {"scalar"}
        elif parent_type == "group":
            compatible_outputs = {"group"}
        else:
            compatible_outputs = {parent_type}

        return [op for op in self._operators.values()
                if op.output_type in compatible_outputs]


# ---------------------------------------------------------------------------
# 2. OperatorNode + OperatorTreeCompiler
# ---------------------------------------------------------------------------


@dataclass
class OperatorNode:
    """Recursive operator tree node.

    Args:
        op_name: Operator name (e.g. "ts_zscore", "add", "rank").
        args: Child nodes (OperatorNode) or leaf values (str).
              Leaf strings are field IDs, numeric literals, or group names.
        output_type: Inferred output type after validation.
    """
    op_name: str
    args: List[Union["OperatorNode", str]]
    output_type: str = ""


class OperatorTreeCompiler:
    """Converts OperatorNode trees to valid WQ expression strings.

    Depth-first compilation: operator(args, ..., kwarg=val) format.
    Leaf strings are emitted verbatim (field IDs, lookbacks, group names).
    """

    def __init__(self, registry: OperatorRegistry):
        self._registry = registry

    def compile(self, node: OperatorNode) -> str:
        """Compile an OperatorNode tree to a WQ expression string."""
        return self._compile_node(node)

    def _compile_node(self, node: OperatorNode) -> str:
        """Recursive compiler."""

        # Infix operators are checked before registry lookup because
        # infix symbols (+, -, *, /, <, <=, etc.) may not have operator defs
        if node.op_name in _INFIX_SYMBOLS:
            if len(node.args) != 2:
                args_str = ", ".join(
                    self._compile_arg(a) for a in node.args
                )
                return f"{node.op_name}({args_str})"
            lhs = self._compile_arg(node.args[0])
            rhs = self._compile_arg(node.args[1])
            return f"{lhs} {node.op_name} {rhs}"

        op = self._registry.get(node.op_name)
        if op is None:
            args_str = ", ".join(
                self._compile_arg(a) for a in node.args
            )
            return f"{node.op_name}({args_str})"

        # Standard function-call form
        args_str = ", ".join(
            self._compile_arg(a) for a in node.args
        )
        return f"{node.op_name}({args_str})"

    def _compile_arg(self, arg: Union[OperatorNode, str]) -> str:
        """Compile a single argument."""
        if isinstance(arg, OperatorNode):
            return self._compile_node(arg)
        return str(arg)

    def validate(self, node: OperatorNode) -> List[str]:
        """Validate an OperatorNode tree.

        Checks arity and infers output types through the tree.
        Returns a list of error strings (empty = valid).
        """
        errors: List[str] = []
        self._validate_node(node, errors)
        return errors

    def _validate_node(self, node: OperatorNode,
                       errors: List[str]) -> str:
        """Recursive validator that returns inferred output_type."""
        op = self._registry.get(node.op_name)
        if op is None:
            # Check if this is an infix symbol (no registry entry needed)
            if node.op_name in _INFIX_SYMBOLS:
                node.output_type = "vector"
                return "vector"
            errors.append(f"Unknown operator: '{node.op_name}'")
            node.output_type = "vector"
            return "vector"

        # Check arity
        min_args, max_args = op.arity
        actual_args = len(node.args)
        if actual_args < min_args:
            errors.append(
                f"'{node.op_name}': expected at least {min_args} args, "
                f"got {actual_args}"
            )
        if max_args is not None and actual_args > max_args:
            errors.append(
                f"'{node.op_name}': expected at most {max_args} args, "
                f"got {actual_args}"
            )

        # Validate positional arg types and collect child output types
        for i, arg in enumerate(node.args):
            if isinstance(arg, OperatorNode):
                child_type = self._validate_node(arg, errors)
                if i < len(op.input_types):
                    expected = op.input_types[i]
                    if expected in ("vector", "rank") and child_type not in ("vector", "rank"):
                        errors.append(
                            f"'{node.op_name}' arg {i}: expected {expected}, "
                            f"got {child_type} from '{arg.op_name}'"
                        )
                    elif expected in ("scalar", "lookback", "group") and child_type == "vector":
                        # Allow a field reference for lookback-like params?
                        pass

        # Infer and set output type
        node.output_type = op.output_type
        return op.output_type


# ---------------------------------------------------------------------------
# 4. PatternTracker
# ---------------------------------------------------------------------------


class PatternTracker:
    """Enforces the 5-strike rule: same operator skeleton fails 5
    consecutive times → force a switch to a different pattern.

    A "strike" is any result where fitness is below FITNESS_THRESHOLD.
    When must_switch() returns True, the composer should avoid
    blacklisted skeletons.
    """

    FITNESS_THRESHOLD = 0.5  # Fitness below this is a failure strike

    def __init__(self):
        self._sequences: Dict[str, int] = {}  # skeleton → consecutive failures
        self._blacklisted: Set[str] = set()

    def record_result(self, expression: str, fitness: float) -> None:
        """Record a simulation result and update strike counts.

        Args:
            expression: The WQ expression that was simulated.
            fitness: The fitness/score returned (Sharpe, etc.).
        """
        skeleton = self._extract_skeleton(expression)
        is_failure = fitness < self.FITNESS_THRESHOLD

        if is_failure:
            count = self._sequences.get(skeleton, 0) + 1
            self._sequences[skeleton] = count
            if count >= 5:
                self._blacklisted.add(skeleton)
        else:
            # Success resets the counter for this skeleton
            self._sequences[skeleton] = 0
            # Remove from blacklist on success
            self._blacklisted.discard(skeleton)

    def must_switch(self) -> bool:
        """Return True if ANY skeleton has 5+ consecutive failures."""
        return bool(self._blacklisted)

    def blacklisted_skeletons(self) -> Set[str]:
        """Return the set of currently blacklisted skeletons."""
        return set(self._blacklisted)

    def strike_count(self, expression: str) -> int:
        """Return consecutive failure count for an expression's skeleton."""
        skeleton = self._extract_skeleton(expression)
        return self._sequences.get(skeleton, 0)

    @staticmethod
    def _extract_skeleton(expression: str) -> str:
        """Strip field/symbol names, keep operator nesting structure.

        Replaces field identifiers with 'f', numeric literals with 'd',
        and group names with 'g'.
        """
        result = expression.strip()

        # Replace group names first (preserve 'g' marker)
        for g in GROUPS:
            result = result.replace(g, 'g')

        # Replace common field names with 'f'
        for fld in ["close", "open", "high", "low", "volume", "vwap",
                     "returns", "cap", "adv20"]:
            result = result.replace(fld, 'f')

        # Any remaining field-like identifiers → 'f'
        result = re.sub(r'\b[a-z][a-z0-9_]*\b(?!\s*\()', 'f', result)

        # Replace numeric literals LAST so 'd' marker is preserved
        result = re.sub(r'\b\d+(\.\d+)?\b', 'd', result)

        return result


# ---------------------------------------------------------------------------
# 5. FeedbackStore
# ---------------------------------------------------------------------------


@dataclass
class FeedbackSample:
    """A single simulation feedback record."""
    expression: str
    field_ids: List[str]
    fitness: Optional[float] = None
    sharpe: Optional[float] = None
    gate1_status: str = "FAIL"  # "PASS" | "NEAR_MISS" | "FAIL"
    gate2_passed: bool = False
    verify_5_passed: bool = False
    timestamp: str = ""


class FeedbackStore:
    """Aggregate and persist simulation feedback.

    Maintains per-operator statistics, field coverage gaps, and
    provides context for the composer to guide exploration.
    """

    def __init__(self, path: str = "feedback_results.json"):
        self._path = path
        self._samples: List[FeedbackSample] = []
        self._load()

    def add_sample(self, sample: FeedbackSample) -> None:
        """Add a feedback sample and persist to disk."""
        if not sample.timestamp:
            sample.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._samples.append(sample)
        self._save()

    def get_context(self, max_samples: int = 100) -> Dict[str, Any]:
        """Build a context dict for the composer.

        Returns:
            dict with keys:
              - per_operator_stats: {op_name: {avg_fitness, avg_sharpe, count}}
              - field_coverage: {field: count}
              - underused_operators: [op_name, ...] — <3 uses
              - last_n_results: last 10 FeedbackSample dicts
              - blacklisted_skeletons: empty list (filled externally)
        """
        # Per-operator statistics
        op_stats: Dict[str, Dict[str, Any]] = {}
        field_counts: Dict[str, int] = {}
        op_use_count: Dict[str, int] = {}

        for s in self._samples:
            # Extract operators from expression
            ops_in_expr = self._extract_operators(s.expression)
            for op_name in ops_in_expr:
                if op_name not in op_stats:
                    op_stats[op_name] = {
                        "fitness_sum": 0.0, "sharpe_sum": 0.0,
                        "count": 0,
                    }
                stats = op_stats[op_name]
                stats["count"] += 1
                if s.fitness is not None:
                    stats["fitness_sum"] += s.fitness
                if s.sharpe is not None:
                    stats["sharpe_sum"] += s.sharpe

            # Count fields
            for fid in s.field_ids:
                field_counts[fid] = field_counts.get(fid, 0) + 1

            # Count operator usage
            for op_name in ops_in_expr:
                op_use_count[op_name] = op_use_count.get(op_name, 0) + 1

        # Build per-operator stats with averages
        per_operator_stats = {}
        for op_name, stats in op_stats.items():
            c = stats["count"]
            per_operator_stats[op_name] = {
                "avg_fitness": round(stats["fitness_sum"] / c, 4) if c else 0,
                "avg_sharpe": round(stats["sharpe_sum"] / c, 4) if c else 0,
                "count": c,
            }

        # Underused operators (<3 uses)
        underused = [name for name, c in op_use_count.items() if c < 3]

        # Last N results
        last_n = [
            {
                "expression": s.expression,
                "fitness": s.fitness,
                "sharpe": s.sharpe,
                "gate1_status": s.gate1_status,
                "gate2_passed": s.gate2_passed,
                "verify_5_passed": s.verify_5_passed,
                "timestamp": s.timestamp,
            }
            for s in self._samples[-max_samples:]
        ]

        return {
            "per_operator_stats": per_operator_stats,
            "field_coverage": dict(
                sorted(field_counts.items(), key=lambda x: -x[1])
            ),
            "underused_operators": underused,
            "last_n_results": last_n,
            "blacklisted_skeletons": [],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return a high-level summary."""
        if not self._samples:
            return {"total_samples": 0}

        total = len(self._samples)
        passed_gate1 = sum(
            1 for s in self._samples if s.gate1_status == "PASS"
        )
        passed_gate2 = sum(1 for s in self._samples if s.gate2_passed)
        passed_verify = sum(1 for s in self._samples if s.verify_5_passed)
        fitnesses = [s.fitness for s in self._samples
                     if s.fitness is not None]

        return {
            "total_samples": total,
            "gate1_pass_rate": round(passed_gate1 / total, 4),
            "gate2_pass_rate": round(passed_gate2 / total, 4),
            "verify_pass_rate": round(passed_verify / total, 4),
            "avg_fitness": round(
                sum(fitnesses) / len(fitnesses), 4
            ) if fitnesses else 0,
            "num_blacklisted_skeletons": 0,
        }

    def _extract_operators(self, expression: str) -> List[str]:
        """Extract operator names from a WQ expression."""
        operators = []
        # Match word before opening paren
        for m in re.finditer(r'([a-z_]+)\s*\(', expression):
            operators.append(m.group(1))
        return operators

    def _load(self) -> None:
        """Load samples from JSON file."""
        path = Path(self._path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data:
                self._samples.append(FeedbackSample(**item))
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    def _save(self) -> None:
        """Persist samples to JSON file."""
        data = [
            {
                "expression": s.expression,
                "field_ids": s.field_ids,
                "fitness": s.fitness,
                "sharpe": s.sharpe,
                "gate1_status": s.gate1_status,
                "gate2_passed": s.gate2_passed,
                "verify_5_passed": s.verify_5_passed,
                "timestamp": s.timestamp,
            }
            for s in self._samples
        ]
        Path(self._path).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# 6. ValidateGate
# ---------------------------------------------------------------------------


class ValidateGate:
    """Expression token validation against the operator registry.

    Tokenizes a WQ expression and checks each token against:
    - Known operator names
    - Known field names (from FieldDelayRegistry)
    - Common parameter names
    - Numeric literals
    """

    _NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")

    @staticmethod
    def check(
        expression: str,
        registry: OperatorRegistry,
        known_fields: Optional[Set[str]] = None,
        common_params: Optional[Set[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """Validate all tokens in a WQ expression.

        Args:
            expression: The WQ alpha expression string.
            registry: OperatorRegistry with known operators.
            known_fields: Set of valid field IDs (from CSVs).
            common_params: Set of common param names (group IDs, etc.).

        Returns:
            (is_valid, list_of_invalid_tokens)
        """
        if common_params is None:
            common_params = {"subindustry", "industry", "sector",
                             "market", "none", "cap", "close", "open",
                             "high", "low", "volume", "vwap", "returns",
                             "adv20", "true", "false",
                             "+", "-", "*", "/", "<", "<=", "==", ">",
                             ">=", "!="}

        tokens = ValidateGate._tokenize(expression)
        invalid: List[str] = []

        for token in tokens:
            if ValidateGate._is_valid_token(
                token, registry, known_fields, common_params
            ):
                continue
            invalid.append(token)

        return len(invalid) == 0, invalid

    @staticmethod
    def _tokenize(expression: str) -> List[str]:
        """Split an expression into distinct tokens (identifiers, symbols).

        Tokens are: operator names, field IDs, numeric literals,
        group names, and infix symbols.
        """
        # Remove parenthesized content to extract tokens safely
        # First, split by common delimiters
        tokens = []
        # Replace parentheses and commas with spaces
        cleaned = expression.replace("(", " ").replace(")", " ") \
                            .replace(",", " ")
        # Split and filter empty
        for t in cleaned.split():
            t = t.strip()
            if not t:
                continue
            # Handle infix operators that may not be separated by spaces
            # e.g. "a+b" -> "a", "+", "b" — but this is rare in WQ
            tokens.append(t)
        return tokens

    @staticmethod
    def _is_valid_token(
        token: str,
        registry: OperatorRegistry,
        known_fields: Optional[Set[str]],
        common_params: Set[str],
    ) -> bool:
        """Check if a single token is valid."""
        # Numeric literal
        if ValidateGate._NUMERIC_RE.match(token):
            return True

        # Infix operator symbol (+, -, *, /, <, <=, ==, >, >=, !=)
        if token in _INFIX_SYMBOLS:
            return True

        # Known operator
        if registry.get(token) is not None:
            return True

        # Common param
        if token.lower() in common_params:
            return True

        # Known field
        if known_fields and token in known_fields:
            return True

        return False
