# ⛔ READ-ONLY — DO NOT MODIFY. Protected engine module.
"""
StateManager — JSON state persistence for the WQ Alpha Pipeline.

Provides crash-recovery via --resume by saving pipeline state after EACH
simulation (never batched). Uses human-readable JSON (never pickle).

Schema (pipeline_state.json):
    {
        "run_id": "uuid4-string",
        "started_at": "ISO-8601",
        "updated_at": "ISO-8601",
        "fields_cache_version": 1,
        "alphas": {
            "<expression_hash>": {
                "alpha_id": "str | None",
                "expression": "str",
                "settings": { ... },
                "metrics": { ... },
                "status": "PASS | NEAR_MISS | HARD_FAIL | CORR_FAIL | SIM_FAIL",
                "pivot_history": [
                    {
                        "parent_alpha_id": "str | None",
                        "strategy": "str",
                        "generation": 1
                    }
                ],
                "simulated_at": "ISO-8601"
            }
        },
        "submitted_ids": ["alpha_id", ...],
        "stats": {
            "total_simulated": 0,
            "pass": 0,
            "near_miss": 0,
            "hard_fail": 0,
            "corr_fail": 0,
            "submitted": 0
        }
    }
"""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Valid status values for alpha records
VALID_STATUSES = frozenset({
    "PASS",
    "NEAR_MISS",
    "HARD_FAIL",
    "CORR_FAIL",
    "SIM_FAIL",
})


def _compute_expression_hash(expression: str, settings: Dict[str, Any]) -> str:
    """Compute a deterministic hash from expression + sorted settings.

    Same expression with different settings produces a different hash.
    Same expression with same settings always produces the same hash.
    """
    # Normalize settings to a sorted dict string for consistent hashing
    normalized = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    raw = f"{expression}|{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _empty_state() -> Dict[str, Any]:
    """Return a fresh pipeline state dict with default values."""
    return {
        "run_id": str(uuid.uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "fields_cache_version": 1,
        "alphas": {},
        "submitted_ids": [],
        "stats": {
            "total_simulated": 0,
            "pass": 0,
            "near_miss": 0,
            "hard_fail": 0,
            "corr_fail": 0,
            "submitted": 0,
        },
    }


class StateManager:
    """Manages pipeline state persistence in a JSON file.

    State is saved after EACH simulation for crash recovery.
    Corrupt JSON files are handled gracefully — a warning is logged
    and a fresh state is returned.

    Args:
        state_path: Path to the JSON state file. Defaults to
            ``pipeline_state.json`` in the current working directory.
    """

    def __init__(self, state_path: str = "pipeline_state.json") -> None:
        self.state_path = state_path
        self._state: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write current state to disk (JSON, human-readable).

        Called after EACH simulation — never batched.
        """
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp_path = self.state_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2, ensure_ascii=False)
            # Atomic-ish write: write to temp then rename
            os.replace(tmp_path, self.state_path)
        except OSError as exc:
            logger.error("Failed to save state to %s: %s", self.state_path, exc)
            # Clean up temp file if rename failed
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def load(self) -> Dict[str, Any]:
        """Reload state from disk. Returns the current state dict.

        Handles corrupt files gracefully: logs a warning and returns
        a fresh empty state.
        """
        self._state = self._load()
        return self._state

    def update_alpha(
        self,
        alpha_id: Optional[str],
        expression: str,
        settings: Dict[str, Any],
        metrics: Dict[str, Any],
        status: str,
        pivot_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Add or update an alpha record in state.

        Args:
            alpha_id: WQ alpha ID (None if simulation failed).
            expression: Alpha expression string.
            settings: Alpha settings dict (decay, neutralization, etc.).
            metrics: Simulation metrics dict (fitness, sharpe, etc.).
            status: One of VALID_STATUSES.
            pivot_history: List of pivot records with parent_alpha_id,
                strategy, and generation.

        Returns:
            The expression hash used as the key in the alphas dict.
        """
        if status not in VALID_STATUSES:
            logger.warning("Unknown status %r, defaulting to SIM_FAIL", status)
            status = "SIM_FAIL"

        expr_hash = _compute_expression_hash(expression, settings)

        # Preserve existing pivot_history if updating and no new history given
        existing = self._state["alphas"].get(expr_hash, {})
        if pivot_history is None:
            pivot_history = existing.get("pivot_history", [])

        self._state["alphas"][expr_hash] = {
            "alpha_id": alpha_id,
            "expression": expression,
            "settings": settings,
            "metrics": metrics,
            "status": status,
            "pivot_history": pivot_history,
            "simulated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Update counters
        self._state["stats"]["total_simulated"] = len(self._state["alphas"])
        self._recount_stats()

        # Save immediately after each simulation
        self.save()

        return expr_hash

    def is_simulated(self, expression: str, settings: Dict[str, Any]) -> bool:
        """Check if an expression+settings combo has already been simulated.

        Used by --resume to skip already-completed simulations.

        Args:
            expression: Alpha expression string.
            settings: Alpha settings dict.

        Returns:
            True if this exact expression+settings combo exists in state.
        """
        expr_hash = _compute_expression_hash(expression, settings)
        return expr_hash in self._state["alphas"]

    def get_stats(self) -> Dict[str, int]:
        """Return summary counters for the current run.

        Returns:
            Dict with keys: total_simulated, pass, near_miss, hard_fail,
            corr_fail, submitted.
        """
        return dict(self._state["stats"])

    def reset(self) -> None:
        """Clear all state for a fresh run."""
        self._state = _empty_state()
        self.save()

    def add_submitted_id(self, alpha_id: str) -> None:
        """Record a successfully submitted alpha ID.

        Args:
            alpha_id: The WQ alpha ID that was submitted.
        """
        if alpha_id not in self._state["submitted_ids"]:
            self._state["submitted_ids"].append(alpha_id)
            self._state["stats"]["submitted"] = len(self._state["submitted_ids"])
            self.save()

    def get_alpha(self, expression: str, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve an alpha record by expression+settings.

        Args:
            expression: Alpha expression string.
            settings: Alpha settings dict.

        Returns:
            The alpha record dict, or None if not found.
        """
        expr_hash = _compute_expression_hash(expression, settings)
        return self._state["alphas"].get(expr_hash)

    def get_all_alphas(self) -> Dict[str, Dict[str, Any]]:
        """Return all alpha records keyed by expression hash."""
        return dict(self._state["alphas"])

    def get_submitted_ids(self) -> List[str]:
        """Return list of all submitted alpha IDs."""
        return list(self._state["submitted_ids"])

    @property
    def run_id(self) -> str:
        """Current run ID."""
        return self._state["run_id"]

    @property
    def started_at(self) -> str:
        """ISO-8601 timestamp of when this run started."""
        return self._state["started_at"]

    @property
    def updated_at(self) -> str:
        """ISO-8601 timestamp of last state update."""
        return self._state["updated_at"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        """Load state from JSON file, handling corrupt files gracefully."""
        if not os.path.exists(self.state_path):
            logger.info("No state file at %s — starting fresh", self.state_path)
            return _empty_state()

        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Corrupt state file at %s (%s) — starting fresh",
                self.state_path,
                exc,
            )
            return _empty_state()
        except OSError as exc:
            logger.warning(
                "Cannot read state file at %s (%s) — starting fresh",
                self.state_path,
                exc,
            )
            return _empty_state()

        # Validate minimum required keys
        required_keys = {"run_id", "alphas", "stats"}
        if not required_keys.issubset(data.keys()):
            logger.warning(
                "State file missing required keys %s — starting fresh",
                required_keys - data.keys(),
            )
            return _empty_state()

        # Ensure all expected keys exist (forward-compatible)
        empty = _empty_state()
        for key in empty:
            data.setdefault(key, empty[key])

        # Ensure stats has all expected counters
        for key in empty["stats"]:
            data["stats"].setdefault(key, 0)

        return data

    def _recount_stats(self) -> None:
        """Recount status counters from the alphas dict.

        Called after every update_alpha to keep stats accurate.
        """
        counts = {
            "pass": 0,
            "near_miss": 0,
            "hard_fail": 0,
            "corr_fail": 0,
        }
        for record in self._state["alphas"].values():
            status = record.get("status", "").upper()
            if status == "PASS":
                counts["pass"] += 1
            elif status == "NEAR_MISS":
                counts["near_miss"] += 1
            elif status == "HARD_FAIL":
                counts["hard_fail"] += 1
            elif status == "CORR_FAIL":
                counts["corr_fail"] += 1
            # SIM_FAIL doesn't increment any status counter

        self._state["stats"]["pass"] = counts["pass"]
        self._state["stats"]["near_miss"] = counts["near_miss"]
        self._state["stats"]["hard_fail"] = counts["hard_fail"]
        self._state["stats"]["corr_fail"] = counts["corr_fail"]
        self._state["stats"]["total_simulated"] = len(self._state["alphas"])
        self._state["stats"]["submitted"] = len(self._state["submitted_ids"])