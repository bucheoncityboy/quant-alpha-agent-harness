# ⛔ READ-ONLY — DO NOT MODIFY. Protected engine module.
"""Reusable WQ simulation runner with gate checking and submission.

Standardizes the simulation → gate check → submit flow duplicated across
all batch scripts. Supports both submit_alpha (from ace_lib) and manual
gate polling patterns.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.ace_lib import start_session, submit_alpha
from engine.rate_limiter import RateLimitedSimulator

logger = logging.getLogger("sim_runner")

# ---------------------------------------------------------------------------
# Gate checking
# ---------------------------------------------------------------------------


def check_gates_manual(
    aid: str,
    session: Any,
    max_wait: int = 60,
) -> Tuple[bool, List[str], Optional[float]]:
    """Manual gate polling via GET /alphas/{aid}.

    Returns (all_pass, fail_reasons, self_corr_value).
    Used by scripts that don't rely on submit_alpha's built-in gate check.
    """
    for wait in [10, 15, 20, 30, max_wait]:
        time.sleep(wait)
        r = session.get(f"https://api.worldquantbrain.com/alphas/{aid}")
        if r.status_code == 200:
            data = r.json()
        elif r.status_code == 429:
            continue
        else:
            continue
        gates = data.get("gates", []) if isinstance(data, dict) else []
        fails: List[str] = []
        sc_val: Optional[float] = None
        for g in gates:
            name = g.get("gateName", "")
            status = g.get("status", "")
            if status != "PASS":
                fails.append(f"{name}={status}")
            if "self" in name.lower():
                try:
                    sc_val = float(str(g.get("limit", "0")).replace("limit=", ""))
                except (ValueError, TypeError):
                    pass
        if fails or sc_val is not None:
            return len(fails) == 0, fails, sc_val
    return False, ["timeout waiting for gates"], None


def check_gates_submit(
    aid: str,
    session: Any,
) -> Tuple[bool, List[str], Optional[float]]:
    """Gate checking via POST /alphas/{aid}/submit + GET polling.

    Returns (all_pass, fail_reasons, self_corr_value).
    Uses submit_alpha pattern from ace_lib.
    """
    r = session.post(f"https://api.worldquantbrain.com/alphas/{aid}/submit")
    if r.headers.get("retry-after"):
        rt = float(r.headers.get("retry-after"))
        time.sleep(rt + 1)
    for _ in range(10):
        r = session.get(f"https://api.worldquantbrain.com/alphas/{aid}/submit")
        if r.text.strip() and len(r.text.strip()) > 50:
            break
        time.sleep(3)
    if not r.text.strip():
        return False, ["empty submit response"], None
    data = r.json()
    checks = data.get("is", {}).get("checks", [])
    fails: List[str] = []
    sc_val: Optional[float] = None
    for c in checks:
        v = c.get("value")
        if c.get("name") == "SELF_CORRELATION":
            sc_val = v
        if c.get("result") != "PASS":
            fails.append(c.get("name"))
    return len(fails) == 0, fails, sc_val


def submit_and_verify(aid: str, session: Any, max_poll: int = 12) -> bool:
    """Submit alpha and poll until status='active'.

    Returns True if active.
    """
    r = session.post(f"https://api.worldquantbrain.com/alphas/{aid}/submit")
    if r.status_code == 429:
        return False
    if r.status_code != 200:
        return False
    for _ in range(max_poll):
        time.sleep(10)
        r2 = session.get(f"https://api.worldquantbrain.com/alphas/{aid}")
        if r2.status_code == 200:
            data = r2.json() if isinstance(r2.json(), dict) else {}
            status = data.get("status", "") if isinstance(data, dict) else ""
            if status == "active":
                return True
    return False


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------


def save_to_submitted(
    submitted: List[Dict[str, Any]],
    submitted_path: Path,
) -> None:
    """Append submitted alphas to the submitted_alphas.json tracker."""
    existing: List[Dict[str, Any]] = []
    if submitted_path.exists():
        existing = json.loads(submitted_path.read_text(encoding="utf-8")).get("alphas", [])
    for sa in submitted:
        existing.append({
            "id": f"wq_{sa['aid']}",
            "alpha_id": sa["aid"],
            "regular": sa["expr"],
            "expression": sa["expr"],
            "fitness": sa.get("fit", 0),
            "sharpe": sa.get("shr", 0),
            "status": "active",
            "date_submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    submitted_path.write_text(
        json.dumps({"version": 1, "count": len(existing), "alphas": existing}, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Complete run pipeline
# ---------------------------------------------------------------------------


def run_simulation_pipeline(
    variants: List[Dict[str, Any]],
    *,
    max_parallel: int = 3,
    max_retries: int = 5,
    gate_wait: int = 60,
    use_manual_gates: bool = True,
    submitted_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Full simulation → phase1 → gate-check → submit pipeline.

    Args:
        variants: List of dicts with keys 'label', 'expr', and optionally 'neut'.
        max_parallel: Parallel simulation limit.
        max_retries: Simulation retry count.
        gate_wait: Seconds to wait for self-correlation computation.
        use_manual_gates: If True, use check_gates_manual; else use check_gates_submit.
        submitted_path: Path to save submitted alphas JSON. Defaults to
                        caller's wq_dir/submitted_alphas.json.

    Returns:
        List of submitted alpha dicts.
    """
    from engine.settings_utils import make_batch_item

    wq_dir = Path.cwd()
    spath = submitted_path or (wq_dir / "submitted_alphas.json")

    session = start_session()
    sim = RateLimitedSimulator(
        session,
        initial_cooldown=0,
        inter_sim_delay=0,
        max_parallel=max_parallel,
        max_retries=max_retries,
        base_backoff=30,
    )

    # Build batch
    sim_data = []
    for v in variants:
        overrides = {}
        if v.get("neut"):
            overrides["neutralization"] = v["neut"]
        sd = make_batch_item(v["expr"], settings_overrides=overrides)
        sim_data.append((v["label"], v["expr"], sd))

    logger.info("=" * 60)
    logger.info("Simulation pipeline: %d variants", len(sim_data))
    logger.info("=" * 60)
    for label, expr, _ in sim_data:
        logger.info("  %s: %s", label, expr[:80])

    batch = [sd for _, _, sd in sim_data]
    results = sim.simulate_batch(batch)

    # Phase 1: collect gate-1 passers
    gate1_passers = []
    for (label, expr, _), res in zip(sim_data, results):
        aid = res.get("alpha_id")
        if not aid:
            logger.warning("%s: no alpha_id", label)
            continue
        metrics = res.get("metrics", {}) if isinstance(res, dict) else {}
        if isinstance(metrics, dict):
            fit = metrics.get("fitness", 0) or 0
            shr = metrics.get("sharpe", 0) or 0
        else:
            fit = shr = 0
        logger.info("%s: fit=%.2f shr=%.2f", label, fit, shr)
        if fit >= 1.0 and shr >= 1.25:
            gate1_passers.append({
                "label": label,
                "aid": aid,
                "expr": expr,
                "fit": fit,
                "shr": shr,
            })
            logger.info("  -> Gate 1 PASS")
        else:
            logger.info("  -> Gate 1 FAIL")

    if not gate1_passers:
        logger.info("No gate-1 passers. Done.")
        return []

    # Phase 2: gate check + submit
    logger.info("\nWaiting %ds for self-correlation computation...", gate_wait)
    time.sleep(gate_wait)

    submitted = []
    for info in gate1_passers:
        aid = info["aid"]
        if use_manual_gates:
            all_pass, fails, sc_val = check_gates_manual(aid, session)
        else:
            all_pass, fails, sc_val = check_gates_submit(aid, session)
        logger.info(
            "%s: sc=%s gates=%s",
            info["label"],
            f"{sc_val:.4f}" if sc_val else "N/A",
            fails if fails else "ALL PASS",
        )
        if all_pass:
            ok = submit_and_verify(aid, session)
            if ok:
                logger.info(
                    "  >>> SUBMITTED & ACTIVE: %s (%s) fit=%.2f shr=%.2f",
                    info["label"], info["aid"], info["fit"], info["shr"],
                )
                submitted.append({**info, "sc": sc_val})
            else:
                logger.warning("  Submit failed for %s", info["label"])
        else:
            logger.info("  Gates FAIL: %s", fails)

    if submitted:
        save_to_submitted(submitted, spath)
        logger.info("\n=== DONE: %d new alphas ACTIVE ===", len(submitted))
        for sa in submitted:
            logger.info(
                "  %s: %s fit=%.2f shr=%.2f sc=%.4f",
                sa["label"], sa["aid"], sa["fit"], sa["shr"], sa.get("sc", 0),
            )

    return submitted
