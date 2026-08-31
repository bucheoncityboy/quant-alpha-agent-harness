"""Unit tests for StateManager."""

import json
import os
import tempfile
import unittest

from engine.state_manager import StateManager


class TestStateManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mktemp(suffix=".json")
        self.sm = StateManager(state_path=self.tmp)

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_initial_state(self):
        """Fresh StateManager has empty stats."""
        stats = self.sm.get_stats()
        self.assertEqual(stats["total_simulated"], 0)
        self.assertEqual(stats["pass"], 0)
        self.assertEqual(stats["submitted"], 0)

    def test_update_alpha(self):
        """update_alpha adds a record and increments stats."""
        expr_hash = self.sm.update_alpha(
            alpha_id="abc123",
            expression="rank(close)",
            settings={"decay": 4},
            metrics={"fitness": 1.5, "sharpe": 1.8},
            status="PASS",
        )
        self.assertIsNotNone(expr_hash)
        self.assertTrue(len(expr_hash) > 0)

        stats = self.sm.get_stats()
        self.assertEqual(stats["total_simulated"], 1)
        self.assertEqual(stats["pass"], 1)

    def test_is_simulated(self):
        """is_simulated returns True for existing expression+settings."""
        self.sm.update_alpha("abc123", "rank(close)", {"decay": 4}, {}, "PASS")
        self.assertTrue(self.sm.is_simulated("rank(close)", {"decay": 4}))
        self.assertFalse(self.sm.is_simulated("rank(close)", {"decay": 5}))
        self.assertFalse(self.sm.is_simulated("zscore(close)", {"decay": 4}))

    def test_save_and_load_persistence(self):
        """State persists to disk and can be reloaded."""
        self.sm.update_alpha("abc123", "rank(close)", {"decay": 4}, {"fitness": 1.5}, "PASS")
        self.sm.save()

        sm2 = StateManager(state_path=self.tmp)
        self.assertTrue(sm2.is_simulated("rank(close)", {"decay": 4}))
        self.assertEqual(sm2.get_stats()["pass"], 1)

    def test_corrupt_file_handling(self):
        """Corrupt JSON file is handled gracefully."""
        with open(self.tmp, "w") as f:
            f.write("INVALID JSON DATA")
        sm = StateManager(state_path=self.tmp)
        stats = sm.get_stats()
        self.assertEqual(stats["total_simulated"], 0)

    def test_add_submitted_id(self):
        """add_submitted_id records submission and updates stats."""
        self.sm.add_submitted_id("alpha999")
        self.assertIn("alpha999", self.sm.get_submitted_ids())
        self.assertEqual(self.sm.get_stats()["submitted"], 1)
        # Duplicate should not be added again
        self.sm.add_submitted_id("alpha999")
        self.assertEqual(self.sm.get_stats()["submitted"], 1)

    def test_get_alpha(self):
        """get_alpha retrieves a specific alpha record."""
        self.sm.update_alpha("abc123", "rank(close)", {"decay": 4}, {"fitness": 1.5}, "PASS")
        record = self.sm.get_alpha("rank(close)", {"decay": 4})
        self.assertIsNotNone(record)
        self.assertEqual(record["alpha_id"], "abc123")
        self.assertEqual(record["status"], "PASS")

        missing = self.sm.get_alpha("zscore(close)", {"decay": 4})
        self.assertIsNone(missing)

    def test_reset(self):
        """reset clears all state."""
        self.sm.update_alpha("abc123", "rank(close)", {"decay": 4}, {}, "PASS")
        self.sm.reset()
        self.assertEqual(self.sm.get_stats()["total_simulated"], 0)

    def test_expression_hash_differentiation(self):
        """Same expression with different settings produces different hash."""
        h1 = self.sm.update_alpha("a1", "rank(close)", {"decay": 4}, {}, "PASS")
        h2 = self.sm.update_alpha("a2", "rank(close)", {"decay": 10}, {}, "PASS")
        self.assertNotEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
