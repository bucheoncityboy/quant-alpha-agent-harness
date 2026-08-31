# ⛔ READ-ONLY — DO NOT MODIFY. Protected engine module.
"""
RateLimitedSimulator — Parallel alpha simulation with ThreadPool (max 3).

Wraps ace_lib.start_simulation() / simulation_progress() directly with
exponential backoff, session refresh, and optional state persistence.
Uses ThreadPoolExecutor to run up to 3 simulations in parallel —
one completes, another starts immediately.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

from . import ace_lib

logger = logging.getLogger(__name__)


class RateLimitedSimulator:
    """Parallel alpha simulator with ThreadPool and rate-limit handling.

    Uses ThreadPoolExecutor (max_workers=3) to run simulations concurrently.
    When one finishes, a new one starts immediately — always keeping up to
    3 in flight.
    """

    def __init__(
        self,
        session: ace_lib.SingleSession,
        initial_cooldown: int = 600,
        inter_sim_delay: int = 75,
        max_retries: int = 10,
        base_backoff: int = 60,
        max_parallel: int = 3,
    ) -> None:
        """Initialize the rate-limited simulator.

        Args:
            session: An authenticated ace_lib SingleSession.
            initial_cooldown: Seconds to wait before first simulation (default 600).
            inter_sim_delay: Seconds to wait between simulations (default 75).
            max_retries: Max retries on 429 rate-limit responses (default 10).
            base_backoff: Base seconds for exponential backoff (default 60).
        """
        self.session = session
        self.initial_cooldown = initial_cooldown
        self.inter_sim_delay = inter_sim_delay
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_parallel = max_parallel

    def simulate_one(
        self,
        sim_data: dict,
        state_mgr: Optional[Any] = None,
    ) -> dict:
        """Simulate a single alpha with rate-limit retry logic.

        Flow:
            1. Session timeout check before every simulation
            2. start_simulation() with 429 exponential backoff + session refresh
            3. simulation_progress() polling
            4. set_alpha_properties() tagging with "pipeline_v1"
            5. Optional state_mgr persistence

        Args:
            sim_data: Simulation data dict (must contain expression/settings keys).
            state_mgr: Optional StateManager instance for crash-recovery persistence.

        Returns:
            dict with keys:
                alpha_id (str | None): WQ alpha ID on success, None on failure.
                sim_data (dict): Original simulation data (passthrough).
                metrics (dict | None): Extracted metrics on success.
                status (str): "PASS" on success.
                error (str | None): Error message on failure.
        """
        # Step 1: Session timeout check before EVERY simulation
        self.session = ace_lib.check_session_and_relogin(self.session)

        # Steps 2-4: Start simulation with 429 retry loop
        resp = None
        for attempt in range(self.max_retries + 1):
            resp = ace_lib.start_simulation(self.session, sim_data)

            if resp.status_code == 429:
                backoff = self.base_backoff * (attempt + 1)
                logger.warning(
                    "429 rate-limited (attempt %d/%d). Backoff %ds, refreshing session.",
                    attempt + 1,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)
                self._refresh_session()
                continue

            break  # Non-429 → exit retry loop

        if resp is None:
            logger.error("Exhausted %d retries (all returned 429)", self.max_retries)
            return {
                "alpha_id": None,
                "sim_data": sim_data,
                "error": f"Exhausted {self.max_retries} retries (all 429)",
            }

        if resp.status_code // 100 != 2:
            logger.error(
                "Simulation start failed: HTTP %d — %s",
                resp.status_code,
                resp.text[:200],
            )
            return {
                "alpha_id": None,
                "sim_data": sim_data,
                "error": f"HTTP {resp.status_code}",
            }

        # Step 5: Poll simulation progress
        result = ace_lib.simulation_progress(self.session, resp)

        # Step 6: Check completion
        if not result["completed"]:
            logger.error("simulation_progress returned not completed: %s", result)
            return {
                "alpha_id": None,
                "sim_data": sim_data,
                "error": "simulation_progress failed",
            }

        # Step 7: Extract alpha_id from result
        alpha_id = result["result"]["id"]

        # Step 8: Tag alpha with pipeline_v1
        ace_lib.set_alpha_properties(self.session, alpha_id, tags=["pipeline_v1"])

        # Step 9: Extract metrics
        metrics = self._extract_metrics(result["result"])

        logger.info(
            "Alpha %s — fitness=%.4f sharpe=%.4f turnover=%.4f returns=%.4f",
            alpha_id,
            metrics["fitness"],
            metrics["sharpe"],
            metrics["turnover"],
            metrics["returns"],
        )

        # Step 10: Inter-simulation delay
        time.sleep(self.inter_sim_delay)

        # Step 11: Persist via state_mgr if provided
        if state_mgr is not None:
            expr = sim_data.get("expression", "")
            settings = sim_data.get("settings", {})
            state_mgr.update_alpha(
                alpha_id=alpha_id,
                expression=expr,
                settings=settings,
                metrics=metrics,
                status="PASS",
            )

        return {
            "alpha_id": alpha_id,
            "sim_data": sim_data,
            "metrics": metrics,
            "status": "PASS",
        }

    def simulate_batch(
        self,
        sim_data_list: list[dict],
        state_mgr: Optional[Any] = None,
    ) -> list[dict]:
        """Simulate alphas in parallel using ThreadPool (max 3 concurrent).

        Each simulation goes through simulate_one() with rate-limit handling.
        Submits all alphas to a ThreadPoolExecutor with max_parallel workers.
        When one alpha completes, the next starts immediately.

        Args:
            sim_data_list: List of simulation data dicts.
            state_mgr: Optional StateManager for after-each persistence.

        Returns:
            List of result dicts, one per alpha in sim_data_list.
        """
        results: list[dict] = []

        # Initial cooldown before any simulation
        if self.initial_cooldown > 0:
            logger.info(
                "Initial cooldown: %ds before processing %d alpha(s)",
                self.initial_cooldown,
                len(sim_data_list),
            )
            time.sleep(self.initial_cooldown)

        # Use ThreadPool for parallel execution (max 3 concurrent)
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_idx = {
                executor.submit(self.simulate_one, sim_data, state_mgr): i
                for i, sim_data in enumerate(sim_data_list)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error("Alpha %d failed with exception: %s", idx + 1, exc)
                    result = {
                        "alpha_id": None,
                        "sim_data": sim_data_list[idx],
                        "error": str(exc),
                    }
                results.append(result)

                # Save state after each completion
                if state_mgr is not None:
                    state_mgr.save()

        return results

    def _refresh_session(self) -> None:
        """Force a fresh session by resetting the singleton."""
        ace_lib.SingleSession._instance = None
        ace_lib.SingleSession._initialized = False
        self.session = ace_lib.start_session()
        logger.info("Session refreshed after rate-limit backoff")

    @staticmethod
    def _extract_metrics(result_dict: dict) -> dict:
        """Extract key simulation metrics from a result dict.

        Args:
            result_dict: The full simulation result dict containing an 'is' key
                         with in-sample statistics.

        Returns:
            dict with keys: fitness, sharpe, turnover, returns.
        """
        is_stats = result_dict.get("is", {})
        return {
            "fitness": float(is_stats.get("fitness", 0)),
            "sharpe": float(is_stats.get("sharpe", 0)),
            "turnover": float(is_stats.get("turnover", 0)),
            "returns": float(is_stats.get("returns", 0)),
        }
