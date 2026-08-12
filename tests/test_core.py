import json
from pathlib import Path
import tempfile
import unittest

from autoresearch.execution import (
    activate_config,
    compare_metrics,
    fingerprint,
    validate_proposal,
    verify_execution,
    write_yaml,
)
from autoresearch.planner import PlannerError, _parse_proposal


class CoreTests(unittest.TestCase):
    def test_proposal_schema_is_strict(self):
        with self.assertRaises(PlannerError):
            _parse_proposal('{"changes": {"learning_rate": 0.4}}')

    def test_proposal_rejects_disallowed_and_duplicate_configs(self):
        current = {"learning_rate": 0.6, "nested": {"value": 1}}
        proposal = {
            "hypothesis": "test",
            "reason": "test",
            "expected_effect": "test",
            "changes": {"learning_rate": 0.4},
        }
        errors, candidate = validate_proposal(
            proposal, current, {"learning_rate": [0.4, 0.6]}, set()
        )
        self.assertEqual(errors, [])
        self.assertEqual(candidate["learning_rate"], 0.4)

        errors, _ = validate_proposal(
            proposal, current, {"learning_rate": [0.4, 0.6]}, {fingerprint(candidate)}
        )
        self.assertIn("repeats", errors[0])

        proposal["changes"] = {"unknown": 1}
        errors, _ = validate_proposal(proposal, current, {}, set())
        self.assertIn("not modifiable", errors[0])

    def test_execution_requires_zero_exit_and_valid_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            path.write_text(json.dumps({"status": "success", "score": 2.0}), encoding="utf-8")
            _, errors = verify_execution(7, False, path, {"metric": "score"})
            self.assertIn("return code 7", errors[0])

            path.write_text(json.dumps({"status": "success"}), encoding="utf-8")
            _, errors = verify_execution(0, False, path, {"metric": "score"})
            self.assertIn("required metric is missing", errors[0])

    def test_comparison_supports_direction_and_tie_breaker(self):
        decision, _ = compare_metrics(
            {"loss": 1.0}, {"loss": 2.0}, {"metric": "loss", "direction": "minimize"}
        )
        self.assertEqual(decision, "accept")
        decision, _ = compare_metrics(
            {"score": 1, "reward": 4},
            {"score": 1, "reward": 3},
            {"metric": "score", "direction": "maximize", "tie_breaker_metric": "reward"},
        )
        self.assertEqual(decision, "accept")

    def test_activate_config_physically_restores_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = root / "before.yaml"
            candidate = root / "candidate.yaml"
            current = root / "current.yaml"
            write_yaml(before, {"learning_rate": 0.6})
            write_yaml(candidate, {"learning_rate": 0.4})
            activate_config(candidate, current)
            restored = activate_config(before, current)
            self.assertEqual(restored, {"learning_rate": 0.6})


if __name__ == "__main__":
    unittest.main()
