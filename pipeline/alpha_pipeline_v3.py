#!/usr/bin/env python3
"""WQ Alpha Researcher — single-shot CLI tool.

No auto-loop. No auto-gates. GJC drives every step.

Usage:
    python alpha_pipeline_v3.py --compose "ts_zscore(close, 60)" [--settings ...]
    python alpha_pipeline_v3.py --simulate --alpha-id <id>
    python alpha_pipeline_v3.py --corr --alpha-id <id>
    python alpha_pipeline_v3.py --check --alpha-id <id>
    python alpha_pipeline_v3.py --submit --alpha-id <id>
    python alpha_pipeline_v3.py --validate "my_expression"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from engine.settings_utils import (
    resolve_settings,
    extract_field_tokens,
    _OPERATORS as SETTINGS_OPERATORS,
    BUILTIN_FIELDS as SETTINGS_BUILTIN,
)
from engine.diverse_generator import (
    CATEGORIES,
    GROUPS,
    CategoryFieldLoader,
    FieldCycler,
)
from engine.alpha_researcher import (
    OperatorRegistry,
    OperatorTreeCompiler,
    OperatorNode,
    PatternTracker,
    FeedbackStore,
    FeedbackSample,
    ValidateGate,
)

logger = logging.getLogger("wq_alpha_researcher")

# Paths
SKILL_DIR = Path(__file__).parent.parent
STATE_DIR = SKILL_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
(STATE_DIR / "output").mkdir(exist_ok=True)
(STATE_DIR / "cache").mkdir(exist_ok=True)
(STATE_DIR / "feedback").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS: Dict[str, Any] = {
    "universe": "TOP3000",
    "decay": 5,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
    "pasteurization": "ON",
    "nanHandling": "ON",
    "dateFrom": "2019-01-01",
    "dateTo": "2023-12-31",
}

KNOWN_UNKNOWN_TOKENS = SETTINGS_OPERATORS | SETTINGS_BUILTIN

# ---------------------------------------------------------------------------
# Core operations — each returns a dict for GJC to inspect
# ---------------------------------------------------------------------------


def validate_expression(expr: str, known_fields: Optional[set] = None) -> Dict:
    """Validate a WQ expression string. Returns {valid, invalid_tokens, errors}."""
    reg = OperatorRegistry()
    valid, tokens = ValidateGate.check(expr, reg, known_fields=known_fields)
    return {
        "expression": expr,
        "valid": valid,
        "invalid_tokens": tokens if not valid else [],
        "operators_detected": [
            op.name for op in reg.list_all()
            if re.search(r'\b' + re.escape(op.name) + r'\b', expr)
        ],
    }


def simulate(
    session,
    expr: str = "",
    alpha_id: Optional[str] = None,
    settings: Optional[Dict] = None,
) -> Dict:
    """Run a single simulation via WQ API. Returns results with stats."""
    if settings is None:
        settings = dict(DEFAULT_SETTINGS)

    from engine.ace_lib import simulate_single_alpha, get_specified_alpha_stats

    try:
        sim_data = {
            "type": "REGULAR",
            "regular": expr or "",
            "settings": {**settings, "visualization": False},
        }
        result = simulate_single_alpha(session, sim_data)
        aid = result.get("alpha_id")
        if not aid:
            return {"expression": expr, "error": "Simulation failed — no alpha_id", "success": False}

        # Fetch fitness/sharpe stats
        stats = get_specified_alpha_stats(session, aid)
        fitness = stats.get("fitness", 0) if stats else None
        sharpe = stats.get("sharpe", 0) if stats else None

        return {
            "expression": expr or result.get("simulate_data", {}).get("regular", ""),
            "alpha_id": aid,
            "fitness": fitness,
            "sharpe": sharpe,
            "settings": settings,
            "success": True,
        }
    except Exception as e:
        return {"expression": expr, "error": str(e), "success": False}


def get_correlation(session, alpha_id: str) -> Dict:
    """Get self-correlation data for an alpha."""
    from engine.ace_lib import get_self_corr

    try:
        df = get_self_corr(session, alpha_id)
        max_corr = 0.0
        if df is not None and not df.empty and df.shape[1] >= 2:
            max_corr = float(df.iloc[:, 1].max())
        return {
            "alpha_id": alpha_id,
            "max_correlation": max_corr,
            "passed_gate2": max_corr < 0.7,
            "records": len(df) if df is not None else 0,
        }
    except Exception as e:
        return {"alpha_id": alpha_id, "error": str(e)}


def check_submission(session, alpha_id: str) -> Dict:
    """Check if an alpha is eligible for submission (WQ 5-gate)."""
    from engine.ace_lib import get_check_submission

    try:
        df = get_check_submission(session, alpha_id)
        if df is None or df.empty:
            return {"alpha_id": alpha_id, "passed": False, "gates": []}
        gates = []
        all_passed = True
        for _, row in df.iterrows():
            name = str(row.get("name", "")).strip()
            result = str(row.get("result", "")).strip()
            passed = result.lower() == "pass"
            gates.append({"name": name, "result": result, "passed": passed})
            if not passed:
                all_passed = False
        return {
            "alpha_id": alpha_id,
            "passed": all_passed,
            "gates": gates,
        }
    except Exception as e:
        return {"alpha_id": alpha_id, "error": str(e)}


def submit_alpha(session, alpha_id: str, submitted_path: Path) -> Dict:
    """Submit an alpha to WQ (make it ACTIVE)."""
    from engine.ace_lib import submit_alpha as wq_submit

    try:
        result = wq_submit(session, alpha_id)
        status = result.get("status", "")
        # Record to submitted_alphas.json
        submitted = []
        if submitted_path.exists():
            with open(submitted_path) as f:
                submitted = json.load(f)
        submitted.append({
            "alpha_id": alpha_id,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        })
        with open(submitted_path, "w") as f:
            json.dump(submitted, f, indent=2, ensure_ascii=False)
        return {"alpha_id": alpha_id, "status": status, "success": True}
    except Exception as e:
        return {"alpha_id": alpha_id, "error": str(e), "success": False}


# ---------------------------------------------------------------------------
# Field loading helper
# ---------------------------------------------------------------------------


def load_fields() -> CategoryFieldLoader:
    """Load all field categories and return the loader."""
    cfl = CategoryFieldLoader()
    cfl.load()
    return cfl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WQ Alpha Researcher — single-shot tool")
    p.add_argument("--compose", type=str, help="Compose an operator tree (WQ expression)")
    p.add_argument("--simulate", action="store_true", help="Run simulation on a composed alpha")
    p.add_argument("--alpha-id", type=str, help="Alpha ID for simulate/corr/check/submit")
    p.add_argument("--corr", action="store_true", help="Check correlation (Gate 2)")
    p.add_argument("--check", action="store_true", help="Check submission eligibility (5-gate)")
    p.add_argument("--submit", action="store_true", help="Submit alpha to WQ (make ACTIVE)")
    p.add_argument("--validate", type=str, help="Validate an expression string")
    p.add_argument("--settings", type=str, help="JSON string for simulation settings")
    p.add_argument("--dry-run", action="store_true", help="Skip actual API calls")
    return p


def main():
    p = build_parser()
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not any([args.compose, args.simulate, args.corr, args.check, args.submit, args.validate]):
        p.print_help()
        return

    reg = OperatorRegistry()
    cfl = load_fields()
    all_fields = set(cfl._all_fields) if hasattr(cfl, "_all_fields") and cfl._all_fields else None
    submitted_path = STATE_DIR / "cache/submitted_alphas.json"

    # --validate
    if args.validate:
        result = validate_expression(args.validate, known_fields=all_fields)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # --compose (compile and validate a manually written expression)
    if args.compose:
        # Parse settings
        settings = dict(DEFAULT_SETTINGS)
        if args.settings:
            try:
                custom = json.loads(args.settings)
                settings.update(custom)
            except json.JSONDecodeError as e:
                print(json.dumps({"error": f"Invalid settings JSON: {e}"}, ensure_ascii=False))
                return

        # Validate
        val_result = validate_expression(args.compose, known_fields=all_fields)
        if not val_result["valid"]:
            print(json.dumps({
                "expression": args.compose,
                "valid": False,
                "invalid_tokens": val_result["invalid_tokens"],
                "message": "Expression contains unrecognized tokens",
            }, indent=2, ensure_ascii=False))
            return

        # Resolve settings if simulate will be run
        sim_settings = settings
        try:
            resolved = resolve_settings(args.compose, settings)
            if resolved:
                sim_settings = resolved
        except Exception:
            pass  # fallback to defaults

        result = {
            "expression": args.compose,
            "valid": True,
            "operators_detected": val_result["operators_detected"],
            "settings": sim_settings,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # For operations that need WQ API, require --alpha-id
    if any([args.simulate, args.corr, args.check, args.submit]):
        if not args.alpha_id:
            print(json.dumps({"error": "--alpha-id is required"}, ensure_ascii=False))
            return

    # All following operations need a live WQ session
    if args.dry_run:
        print(json.dumps({"dry_run": True, "message": "Skipping API call"}, ensure_ascii=False))
        return

    from engine.ace_lib import start_session, check_session_and_relogin

    session = start_session()

    # --simulate
    if args.simulate:
        sim_settings = dict(DEFAULT_SETTINGS)
        if args.settings:
            try:
                custom = json.loads(args.settings)
                sim_settings.update(custom)
            except json.JSONDecodeError:
                pass
        result = simulate(session, "", args.alpha_id, sim_settings)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return

    # --corr
    if args.corr:
        result = get_correlation(session, args.alpha_id)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return

    # --check
    if args.check:
        result = check_submission(session, args.alpha_id)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return

    # --submit
    if args.submit:
        result = submit_alpha(session, args.alpha_id, submitted_path)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return


if __name__ == "__main__":
    main()
