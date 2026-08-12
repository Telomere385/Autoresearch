import io
import json
from pathlib import Path
import tempfile
import textwrap
import time
import unittest

from autoresearch.agent import _initial_state, _validate_runtime_config, run_agent
from autoresearch.execution import compare_metrics, load_yaml, verify_execution, write_yaml
from autoresearch.planner import PlannerError, _normalize_assistant_message, parse_tool_call
from autoresearch.progress import ProgressReporter, tool_summary
from autoresearch.report import validate_report
from autoresearch.tooling import ToolHarness


class CoreTests(unittest.TestCase):
    def test_progress_reporter_heartbeats_and_never_prints_file_content(self):
        stream = io.StringIO()
        reporter = ProgressReporter("normal", stream=stream, heartbeat_interval=0.01)
        reporter.emit("FILE", "read task/config.yaml lines 1-10")
        with reporter.waiting("LLM", "call 1 waiting"):
            time.sleep(0.025)
        output = stream.getvalue()
        self.assertIn("FILE", output)
        self.assertIn("LLM", output)
        self.assertIn("elapsed", output)
        summary = tool_summary("write_file", {"path": "run/config.yaml", "content": "SECRET"})
        self.assertEqual(summary, "run/config.yaml")
        self.assertNotIn("SECRET", summary)

        quiet_stream = io.StringIO()
        ProgressReporter("quiet", stream=quiet_stream).emit("RUN", "hidden")
        self.assertEqual(quiet_stream.getvalue(), "")

    def test_native_tool_call_parsing_and_reasoning_replay(self):
        message = _normalize_assistant_message({
            "content": None,
            "reasoning_content": "inspect first",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"goal.md"}'},
            }],
        })
        self.assertEqual(message["reasoning_content"], "inspect first")
        call_id, name, arguments = parse_tool_call(message["tool_calls"][0])
        self.assertEqual((call_id, name), ("call_1", "read_file"))
        self.assertEqual(arguments, {"path": "goal.md"})
        with self.assertRaises(PlannerError):
            parse_tool_call({"id": "bad", "function": {"name": "x", "arguments": "{"}})

    def test_runtime_requires_three_iterations_and_native_llm(self):
        config = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "autoresearch.yaml")
        config["budget"]["min_iterations"] = 2
        with self.assertRaisesRegex(ValueError, "at least 3"):
            _validate_runtime_config(config)
        config["budget"]["min_iterations"] = 3
        config["planner"]["provider"] = "local_command"
        with self.assertRaisesRegex(ValueError, "openai_compatible"):
            _validate_runtime_config(config)

    def test_execution_verification_and_comparison_are_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            path.write_text(json.dumps({"status": "success", "score": 2.0}), encoding="utf-8")
            _, errors = verify_execution(7, False, path, {"metric": "score"})
            self.assertIn("return code 7", errors[0])
        decision, _ = compare_metrics(
            {"score": 2.0}, {"score": 3.0}, {"metric": "score", "direction": "maximize"}
        )
        self.assertEqual(decision, "reject")

    def test_report_validator_rejects_missing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.md"
            report.write_text("# Best Result\nscore 7.0", encoding="utf-8")
            state = {
                "iteration": 3, "objective": {"metric": "score"},
                "best_metrics": {"score": 7.0}, "recovery_events": [],
            }
            errors = validate_report(report, state)
            self.assertTrue(any("Research Goal" in error for error in errors))
            self.assertTrue(any("Iteration 1" in error for error in errors))

    def test_harness_rejects_out_of_run_writes_and_early_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "runs" / "safe_run"
            run_dir.mkdir(parents=True)
            (root / "task").mkdir()
            (root / "configs").mkdir()
            config = _integration_config()
            state = _initial_state("safe_run", "test goal", config)
            state["plan"] = {"steps": [{"id": "work", "status": "completed"}]}
            harness = ToolHarness(root, run_dir, config, state)
            call_dir = run_dir / "tool_calls" / "call_001"
            call_dir.mkdir(parents=True)
            result = harness.execute(
                "write_file", {"path": "task/forbidden.txt", "content": "no"}, call_dir
            )
            self.assertFalse(result["ok"])
            self.assertFalse((root / "task" / "forbidden.txt").exists())
            state["tool_call_count"] = 2
            result = harness.execute(
                "finish", {"summary": "too early", "report_path": "runs/safe_run/report.md"}, call_dir
            )
            self.assertFalse(result["ok"])
            self.assertIn("At least 3", result["error"])


class AgentIntegrationTests(unittest.TestCase):
    def test_llm_plans_and_uses_real_file_and_shell_tools_for_three_iterations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "task").mkdir()
            (root / "runs").mkdir()
            (root / "task" / "goal.md").write_text("Improve score with real experiments.", encoding="utf-8")
            script = root / "task" / "experiment.py"
            script.write_text(textwrap.dedent("""
                import argparse
                import json
                from pathlib import Path
                import yaml

                parser = argparse.ArgumentParser()
                parser.add_argument("--config", required=True)
                parser.add_argument("--run-id", required=True)
                parser.add_argument("--timeout-seconds")
                args = parser.parse_args()
                config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
                output = Path.cwd() / "runs" / args.run_id / "summary.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                score = float(config["learning_rate"]) * 10
                output.write_text(json.dumps({"status": "success", "score": score}), encoding="utf-8")
                print(json.dumps({"status": "success", "score": score}))
            """).strip() + "\n", encoding="utf-8")
            config = _integration_config()
            config_path = root / "configs" / "runtime.yaml"
            write_yaml(config_path, config)

            run_id = "integration"
            run_prefix = f"runs/{run_id}"
            baseline = "learning_rate: 0.5\n"
            candidate_1 = "learning_rate: 0.4\n"
            candidate_2 = "learning_rate: 0.6\n"
            candidate_3 = "learning_rate: 0.7\n"
            plan_steps = [
                {"id": "inspect", "description": "Inspect goal and runtime configuration", "expected_tools": ["read_file"], "success_signal": "Inputs understood"},
                {"id": "baseline", "description": "Write and run baseline", "expected_tools": ["write_file", "run_command"], "success_signal": "Valid baseline metric"},
                {"id": "experiments", "description": "Run at least three candidates", "expected_tools": ["write_file", "run_command", "evaluate_result"], "success_signal": "Three evaluated candidates"},
                {"id": "recovery", "description": "Recover from a real negative result", "expected_tools": ["restore_snapshot"], "success_signal": "Verified rollback"},
                {"id": "report", "description": "Write and validate final report", "expected_tools": ["write_file", "finish"], "success_signal": "Report accepted"},
            ]
            report = _complete_report()
            actions = [
                ("submit_plan", {"goal_summary": "Improve score through measured iterations", "steps": plan_steps, "risks": ["candidate may regress"]}),
                ("read_file", {"path": "configs/runtime.yaml", "start_line": 1, "max_lines": 200}),
                ("update_plan", _plan_update("inspect", "Inputs inspected", "Run baseline")),
                ("write_file", {"path": f"{run_prefix}/current_config.yaml", "content": baseline}),
                ("run_command", _run_args(run_id, "baseline", 0, "Measure baseline")),
                ("update_plan", _plan_update("baseline", "Baseline score is 5.0", "Run candidates")),
                ("write_file", {"path": f"{run_prefix}/current_config.yaml", "content": candidate_1}),
                ("run_command", _run_args(run_id, "candidate", 1, "Test lower learning rate")),
                ("evaluate_result", {"iteration": 1, "decision": "rollback", "reason": "Score declined", "evidence": "4.0 is below 5.0"}),
                ("restore_snapshot", {"snapshot_id": "iteration_01_before.yaml", "reason": "Restore after measured regression"}),
                ("update_plan", _plan_update("recovery", "Iteration 1 rollback verified", "Try a different candidate")),
                ("write_file", {"path": f"{run_prefix}/current_config.yaml", "content": candidate_2}),
                ("run_command", _run_args(run_id, "candidate", 2, "Test higher learning rate")),
                ("evaluate_result", {"iteration": 2, "decision": "accept", "reason": "Score improved", "evidence": "6.0 is above 5.0"}),
                ("write_file", {"path": f"{run_prefix}/current_config.yaml", "content": candidate_3}),
                ("run_command", _run_args(run_id, "candidate", 3, "Test another higher rate")),
                ("evaluate_result", {"iteration": 3, "decision": "accept", "reason": "Score improved again", "evidence": "7.0 is above 6.0"}),
                ("update_plan", _plan_update("experiments", "Three candidates evaluated", "Write the report")),
                ("write_file", {"path": f"{run_prefix}/report.md", "content": report}),
                ("update_plan", _plan_update("report", "Report written with all required evidence", "Finish")),
                ("finish", {"summary": "Completed three real candidates with a verified rollback", "report_path": f"{run_prefix}/report.md"}),
            ]
            fake = ScriptedLLM(actions)
            progress_stream = io.StringIO()
            reporter = ProgressReporter(
                "normal", stream=progress_stream, heartbeat_interval=0.01
            )
            result = run_agent(
                config_path, run_id, "Improve the validation score in bounded experiments.",
                chat_fn=fake, root=root, runs_root=root / "runs", reporter=reporter,
            )

            self.assertEqual(result["status"], "completed")
            state = json.loads((root / "runs" / run_id / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["iteration"], 3)
            self.assertEqual(len(state["history"]), 3)
            self.assertEqual(state["history"][0]["decision"], "rollback")
            self.assertTrue(state["recovery_events"][0]["qualifies"])
            self.assertEqual(state["best_metrics"]["score"], 7.0)
            self.assertTrue(all(state["requirements"].values()))
            self.assertTrue((root / "runs" / run_id / "tool_calls").is_dir())
            self.assertIn('"event": "tool_call"', (root / "runs" / run_id / "trajectory.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(fake.reasoning_was_replayed)
            progress = progress_stream.getvalue()
            for event in ("LLM", "TOOL", "FILE", "EXP", "CHECK", "RECOVER", "DONE"):
                self.assertIn(event, progress)


class ScriptedLLM:
    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0
        self.reasoning_was_replayed = False

    def __call__(self, messages, _tools, _config, tool_choice):
        if self.index and any(message.get("reasoning_content") == "choose a real tool" for message in messages):
            self.reasoning_was_replayed = True
        if tool_choice != "required" or self.index >= len(self.actions):
            raise AssertionError("Unexpected LLM call")
        name, arguments = self.actions[self.index]
        self.index += 1
        message = {
            "role": "assistant", "content": "", "reasoning_content": "choose a real tool",
            "tool_calls": [{
                "id": f"call_{self.index}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }],
        }
        return message, {"choices": [{"message": message}], "usage": {"total_tokens": 1}}


def _integration_config():
    return {
        "goal_file": "task/goal.md",
        "planner": {"provider": "openai_compatible", "model": "test-model"},
        "budget": {
            "min_iterations": 3, "max_iterations": 5, "max_tool_calls": 30,
            "max_consecutive_failures": 3, "max_wall_time_minutes": 2,
            "experiment_timeout_seconds": 10,
        },
        "experiment": {
            "command": ["{python}", "task/experiment.py"],
            "metrics_path": "runs/{experiment_id}/summary.json",
        },
        "objective": {"metric": "score", "direction": "maximize", "target": 7.0},
        "baseline": {"learning_rate": 0.5},
        "search_space": {"learning_rate": [0.4, 0.5, 0.6, 0.7]},
        "evidence": {"require_recovery_event": True},
        "tools": {
            "read_roots": ["task", "configs", "runs/{run_id}"],
            "write_root": "runs/{run_id}", "allowed_commands": ["python task/experiment.py"],
            "max_read_lines": 500, "max_output_chars": 12000,
        },
    }


def _run_args(run_id, kind, iteration, hypothesis):
    label = "baseline" if kind == "baseline" else f"iteration_{iteration:02d}"
    return {
        "argv": [
            "python", "task/experiment.py", "--config", f"runs/{run_id}/current_config.yaml",
            "--run-id", f"{run_id}/{label}/raw", "--timeout-seconds", "10",
        ],
        "timeout_seconds": 10, "experiment_kind": kind,
        "hypothesis": hypothesis, "expected_effect": "Score should provide measured evidence",
    }


def _plan_update(step_id, evidence, next_action):
    return {"step_id": step_id, "status": "completed", "evidence": evidence, "next_action": next_action}


def _complete_report():
    return """# Research Goal
Improve the measured score through bounded, real subprocess experiments.

## Plan
Inspect inputs, establish a baseline, run three candidates, recover from regression, and report evidence.

## Baseline
The baseline score was 5.0 using learning_rate 0.5.

## Experiment Process
Iteration 1 used learning_rate 0.4 and measured score 4.0, so it was rejected.
Iteration 2 used learning_rate 0.6 and measured score 6.0, so it was accepted.
Iteration 3 used learning_rate 0.7 and measured score 7.0, so it was accepted.

## Failure and Recovery
Iteration 1 was a real metric regression. The Agent requested rollback and the harness restored and verified iteration_01_before.yaml before continuing.

## Best Result
The best measured score was 7.0 from Iteration 3.

## Limitations
This small demonstration uses one deterministic metric and a bounded search space, so it is not a statistical performance claim.
"""


if __name__ == "__main__":
    unittest.main()
