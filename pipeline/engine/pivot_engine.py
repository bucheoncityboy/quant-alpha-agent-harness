# ⛔ READ-ONLY — DO NOT MODIFY. Protected engine module.
"""
pivot_engine.py — Token-level WQ BRAIN expression pivots.

PivotEngine parses alpha expressions into typed tokens, then applies small,
validated structural changes to reduce self/prod correlation without unsafe
substring replacement.  In particular, operator mutations happen on exact
operator tokens, so ``rank`` never matches inside ``group_rank`` or ``ts_rank``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

GROUP_TOKENS = {"subindustry", "industry", "sector", "market", "none"}
PRICE_VOLUME_FIELDS = {"cap", "close", "open", "high", "low", "volume", "vwap", "returns", "adv20"}
INFIX_OPERATORS = {"+", "-", "*", "/", ">", "<", ">=", "<=", "==", "!="}
VALID_OPERATORS = {
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
COMMON_PARAMS = GROUP_TOKENS | PRICE_VOLUME_FIELDS
TS_OPERATORS = {
    "days_from_last_change", "hump", "kth_element", "last_diff_value",
    "ts_arg_max", "ts_arg_min", "ts_av_diff", "ts_backfill", "ts_corr",
    "ts_count_nans", "ts_covariance", "ts_decay_linear", "ts_delay", "ts_delta",
    "ts_mean", "ts_product", "ts_quantile", "ts_rank", "ts_regression",
    "ts_scale", "ts_std_dev", "ts_step", "ts_sum", "ts_zscore",
}


@dataclass(frozen=True)
class Token:
    """A typed expression token."""

    type: str
    value: str


@dataclass(frozen=True)
class _Call:
    op_index: int
    open_index: int
    close_index: int
    depth: int


class PivotEngine:
    """Token-level pivot engine for WQ BRAIN FASTEXPR alpha expressions."""

    MAX_PIVOTS = 3

    FAILURE_PRIORITIES: Dict[str, Tuple[str, ...]] = {
        "SELF_CORR_FAIL": (
            "transformation_swap",
            "diversifying_leg",
            "time_horizon_change",
            "grouping_change",
            "field_substitution",
        ),
        "PROD_CORR_FAIL": (
            "field_substitution",
            "transformation_swap",
            "diversifying_leg",
            "time_horizon_change",
            "grouping_change",
        ),
    }

    _TOKEN_RE = re.compile(
        r"[A-Za-z_][A-Za-z0-9_/]*|-?\d+(?:\.\d+)?|>=|<=|==|!=|[(),+\-*<>/]"
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_expression(self, expression: str) -> List[Token]:
        """Parse an expression into typed tokens.

        Token types are ``OPERATOR``, ``FIELD``, ``NUMBER``, ``GROUP``,
        ``PAREN``, and ``COMMA``.  Arithmetic/comparison symbols are emitted
        as ``OPERATOR`` tokens so the parser remains token-level for complete
        FASTEXPR expressions.
        """
        tokens: List[Token] = []
        for raw in self._TOKEN_RE.findall(expression):
            if raw in ("(", ")"):
                tokens.append(Token("PAREN", raw))
            elif raw == ",":
                tokens.append(Token("COMMA", raw))
            elif raw in INFIX_OPERATORS:
                tokens.append(Token("OPERATOR", raw))
            elif self._is_number(raw):
                tokens.append(Token("NUMBER", raw))
            else:
                lowered = raw.lower()
                if lowered in GROUP_TOKENS:
                    tokens.append(Token("GROUP", raw))
                elif raw in VALID_OPERATORS:
                    tokens.append(Token("OPERATOR", raw))
                elif raw in COMMON_PARAMS or raw in PRICE_VOLUME_FIELDS:
                    tokens.append(Token("FIELD", raw))
                else:
                    tokens.append(Token("FIELD", raw))
        return tokens

    def apply(
        self,
        strategy_name: str,
        expression: str,
        settings: Dict[str, Any],
        field_registry: Any,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Apply a pivot strategy or failure-priority recipe.

        Args:
            strategy_name: One strategy method name, or ``SELF_CORR_FAIL`` /
                ``PROD_CORR_FAIL`` to apply the corresponding priority order.
            expression: Original alpha expression.
            settings: Alpha simulation settings copied onto each result.
            field_registry: Object exposing ``validate_expression`` and, for
                field substitution, ``get_alternatives``.

        Returns:
            Up to three ``(pivoted_expression, pivoted_settings)`` tuples.
        """
        strategy_key = strategy_name.upper()
        if strategy_key in self.FAILURE_PRIORITIES:
            strategy_order = self.FAILURE_PRIORITIES[strategy_key]
        else:
            strategy_order = (strategy_name,)

        results: List[Tuple[str, Dict[str, Any]]] = []
        seen = {expression}
        for name in strategy_order:
            method = getattr(self, name, None)
            if method is None or not callable(method):
                raise ValueError(f"Unknown pivot strategy: {strategy_name}")

            for candidate in method(expression, settings, field_registry):
                if candidate in seen:
                    continue
                seen.add(candidate)
                is_valid, _unknowns = field_registry.validate_expression(candidate)
                if not is_valid:
                    continue
                results.append((candidate, dict(settings)))
                if len(results) >= self.MAX_PIVOTS:
                    return results
        return results

    # ------------------------------------------------------------------
    # Strategy methods
    # ------------------------------------------------------------------

    def transformation_swap(
        self, expression: str, settings: Dict[str, Any] | None = None, field_registry: Any | None = None
    ) -> List[str]:
        """Replace exact top-level transformation operator tokens."""
        tokens = self.parse_expression(expression)
        pivots: List[str] = []
        swap_map: Dict[str, Tuple[str, ...]] = {
            "rank": ("zscore", "group_zscore"),
            "zscore": ("rank", "group_zscore"),
            "group_zscore": ("group_rank", "zscore", "rank"),
            "group_rank": ("group_zscore",),
            "ts_rank": ("ts_zscore",),
            "ts_zscore": ("ts_rank",),
        }

        for call in self._top_level_calls(tokens):
            old_op = tokens[call.op_index].value
            for new_op in swap_map.get(old_op, ()):  # exact token lookup only
                mutated = self._mutate_call_operator(tokens, call, new_op)
                if mutated:
                    pivots.append(self._render(mutated))
            if pivots:
                break
        return self._unique(pivots)

    def diversifying_leg(
        self, expression: str, settings: Dict[str, Any] | None = None, field_registry: Any | None = None
    ) -> List[str]:
        """Add proven decorrelation terms to the expression."""
        base = expression.strip()
        return [
            f"{base} + group_zscore(-ts_delta(close, 5), subindustry)",
            f"{base} + group_zscore(-ts_delta(close, 10), subindustry)",
            f"{base} * trade_when(volume > ts_mean(volume, 20), 1, -1)",
        ]

    def time_horizon_change(
        self, expression: str, settings: Dict[str, Any] | None = None, field_registry: Any | None = None
    ) -> List[str]:
        """Double or halve numeric lookback parameters inside TS operators."""
        tokens = self.parse_expression(expression)
        horizon_map = {"10": "20", "20": "40", "30": "60", "60": "30"}
        pivots: List[str] = []

        for call in self._calls(tokens):
            if tokens[call.op_index].value not in TS_OPERATORS:
                continue
            for idx in range(call.open_index + 1, call.close_index):
                tok = tokens[idx]
                if tok.type == "NUMBER" and tok.value in horizon_map:
                    mutated = self._replace_at(tokens, idx, horizon_map[tok.value])
                    pivots.append(self._render(mutated))
        return self._unique(pivots)

    def grouping_change(
        self, expression: str, settings: Dict[str, Any] | None = None, field_registry: Any | None = None
    ) -> List[str]:
        """Change group tokens along SUBINDUSTRY → INDUSTRY → MARKET → NONE."""
        tokens = self.parse_expression(expression)
        group_order = ["subindustry", "industry", "market", "none"]
        pivots: List[str] = []

        for idx, tok in enumerate(tokens):
            if tok.type != "GROUP":
                continue
            current = tok.value.lower()
            if current not in group_order:
                continue
            replacements = [grp for grp in group_order if grp != current]
            start = group_order.index(current)
            replacements = group_order[start + 1 :] + group_order[:start]
            for replacement in replacements:
                mutated = self._replace_at(tokens, idx, replacement)
                pivots.append(self._render(mutated))
            break
        return self._unique(pivots)

    def field_substitution(
        self, expression: str, settings: Dict[str, Any] | None = None, field_registry: Any | None = None
    ) -> List[str]:
        """Replace field tokens with same-category alternatives from registry."""
        if field_registry is None or not hasattr(field_registry, "get_alternatives"):
            return []

        tokens = self.parse_expression(expression)
        pivots: List[str] = []
        queried_fields: set[str] = set()

        for idx, tok in enumerate(tokens):
            if tok.type != "FIELD" or tok.value in queried_fields:
                continue
            queried_fields.add(tok.value)
            for alternative in field_registry.get_alternatives(tok.value):
                if not alternative or alternative == tok.value:
                    continue
                mutated = self._replace_at(tokens, idx, alternative)
                pivots.append(self._render(mutated))
        return self._unique(pivots)

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_number(value: str) -> bool:
        return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", value))

    @staticmethod
    def _replace_at(tokens: Sequence[Token], index: int, value: str) -> List[Token]:
        mutated = list(tokens)
        mutated[index] = Token(tokens[index].type, value)
        return mutated

    @staticmethod
    def _unique(expressions: Iterable[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for expr in expressions:
            if expr not in seen:
                seen.add(expr)
                out.append(expr)
        return out

    def _calls(self, tokens: Sequence[Token]) -> List[_Call]:
        calls: List[_Call] = []
        depth = 0
        for idx, tok in enumerate(tokens[:-1]):
            if tok.value == "(":
                depth += 1
                continue
            if tok.value == ")":
                depth = max(0, depth - 1)
                continue
            if tok.type == "OPERATOR" and tok.value not in INFIX_OPERATORS and tokens[idx + 1].value == "(":
                close_index = self._find_matching_paren(tokens, idx + 1)
                if close_index is not None:
                    calls.append(_Call(idx, idx + 1, close_index, depth))
        return calls

    def _top_level_calls(self, tokens: Sequence[Token]) -> List[_Call]:
        return [call for call in self._calls(tokens) if call.depth == 0]

    @staticmethod
    def _find_matching_paren(tokens: Sequence[Token], open_index: int) -> int | None:
        depth = 0
        for idx in range(open_index, len(tokens)):
            if tokens[idx].value == "(":
                depth += 1
            elif tokens[idx].value == ")":
                depth -= 1
                if depth == 0:
                    return idx
        return None

    def _mutate_call_operator(self, tokens: Sequence[Token], call: _Call, new_op: str) -> List[Token] | None:
        old_op = tokens[call.op_index].value
        mutated = list(tokens)
        mutated[call.op_index] = Token("OPERATOR", new_op)

        if new_op == "group_zscore" and old_op in {"rank", "zscore"}:
            mutated[call.close_index:call.close_index] = [Token("COMMA", ","), Token("GROUP", "subindustry")]
            return mutated

        if old_op == "group_zscore" and new_op in {"rank", "zscore"}:
            removable_start = self._last_top_level_comma(mutated, call.open_index, call.close_index)
            if removable_start is None:
                return None
            del mutated[removable_start:call.close_index]
            return mutated

        return mutated

    @staticmethod
    def _last_top_level_comma(tokens: Sequence[Token], open_index: int, close_index: int) -> int | None:
        depth = 0
        last_comma: int | None = None
        for idx in range(open_index + 1, close_index):
            if tokens[idx].value == "(":
                depth += 1
            elif tokens[idx].value == ")":
                depth -= 1
            elif tokens[idx].type == "COMMA" and depth == 0:
                last_comma = idx
        return last_comma

    def _render(self, tokens: Sequence[Token]) -> str:
        parts: List[str] = []
        prev: Token | None = None

        for idx, tok in enumerate(tokens):
            value = tok.value
            if value == "(":
                parts.append("(")
            elif value == ")":
                self._rstrip_space(parts)
                parts.append(")")
            elif tok.type == "COMMA":
                self._rstrip_space(parts)
                parts.append(", ")
            elif value in INFIX_OPERATORS:
                if value == "-" and self._is_unary_minus(prev, tokens, idx):
                    self._rstrip_space(parts)
                    parts.append("-")
                else:
                    self._rstrip_space(parts)
                    parts.append(f" {value} ")
            else:
                parts.append(value)
            prev = tok

        return "".join(parts).strip()

    @staticmethod
    def _rstrip_space(parts: List[str]) -> None:
        if parts and parts[-1].endswith(" "):
            parts[-1] = parts[-1].rstrip()

    @staticmethod
    def _is_unary_minus(prev: Token | None, tokens: Sequence[Token], index: int) -> bool:
        if tokens[index].value != "-":
            return False
        if prev is None:
            return True
        return prev.value in {"(", ","} or prev.value in INFIX_OPERATORS


__all__ = ["PivotEngine", "Token"]
