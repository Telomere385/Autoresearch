"""Unit and end-to-end tests for the AutoResearch agent harness."""

import io
import json
from pathlib import Path
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

from openai.types.chat import ChatCompletion

from autoresearch.agent import (
    _build_model_context, _compact_tool_result, _initial_state, _state_summary,
    _validate_runtime_config, run_agent,
)
from autoresearch.execution import compare_metrics, load_yaml, verify_execution, write_yaml
from autoresearch.planner import PlannerError, _normalize_assistant_message, chat, parse_tool_call
from autoresearch.progress import ProgressReporter, tool_summary
from autoresearch.report import validate_report
from autoresearch.tooling import ToolHarness


VALID_REPORT = """# Research Goal
Improve the score through bounded, real experiments.

## Plan
Measure a baseline, run three candidates, recover from regression, and report evidence.

## Baseline
The baseline score was 5.0.

## Experiment Process
Iteration 1 scored 4.0 and was rejected. Iteration 2 scored 6.0 and was accepted.
Iteration 3 scored 7.0 and was accepted.

## Failure and Recovery
Iteration 1 regressed; snapshot rollback and recovery were verified before continuing.

## Best Result
The best score was 7.0 from Iteration 3.

## Limitations
This bounded single-seed experiment is not a statistical performance claim.
"""


class CoreTests(unittest.TestCase):
    """Verify planner transport, validation, progress, and safety boundaries."""

    def test_openai_sdk_transport_preserves_kimi_fields_and_configuration(self):
        """Ensure SDK serialization preserves provider-specific reasoning data."""
        payload = ChatCompletion.model_validate({
            "id": "test", "object": "chat.completion", "created": 0,
            "model": "kimi/kimi-k3", "choices": [{
                "index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None,
                    "reasoning_content": "inspect before acting",
                    "tool_calls": [{
                        "id": "call_sdk", "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"goal.md"}'},
                    }],
                },
            }],
            "usage": {"completion_tokens": 5, "prompt_tokens": 7, "total_tokens": 12},
        }).model_dump(mode="json")
        factory = FakeOpenAIClientFactory(payload)
        config = {
            "provider": "openai_compatible", "model": "kimi/kimi-k3",
            "api_key_env": "TEST_OPENAI_KEY", "base_url": "https://example.test/v1/",
            "timeout_seconds": 17, "max_retries": 4, "temperature": 0.1,
        }
        with patch.dict("os.environ", {"TEST_OPENAI_KEY": "secret"}):
            message, raw = chat(
                [{"role": "user", "content": "goal"}], [{"type": "function"}],
                config, client_factory=factory,
            )
        self.assertEqual(message["reasoning_content"], "inspect before acting")
        self.assertEqual(message["tool_calls"][0]["id"], "call_sdk")
        self.assertEqual(raw["usage"]["total_tokens"], 12)
        self.assertEqual(factory.client_kwargs["base_url"], "https://example.test/v1")
        self.assertEqual(factory.client_kwargs["timeout"], 17.0)
        self.assertEqual(factory.client_kwargs["max_retries"], 4)
        self.assertNotIn("secret", json.dumps(factory.create_kwargs))
        self.assertFalse(factory.create_kwargs["parallel_tool_calls"])
        self.assertEqual(factory.create_kwargs["tool_choice"], "required")

    def test_progress_reporter_heartbeats_and_never_prints_file_content(self):
        """Ensure progress remains live without leaking written file contents."""
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
        """Validate native tool arguments and reasoning-content replay fields."""
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
        """Enforce the minimum experiment count and supported LLM provider."""
        config = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "autoresearch.yaml")
        config["budget"]["min_iterations"] = 2
        with self.assertRaisesRegex(ValueError, "at least 3"):
            _validate_runtime_config(config)
        config["budget"]["min_iterations"] = 3
        config["planner"]["provider"] = "local_command"
        with self.assertRaisesRegex(ValueError, "openai_compatible"):
            _validate_runtime_config(config)

    def test_context_configuration_requires_valid_types_and_positive_limits(self):
        """Reject ambiguous context controls before an agent run starts."""
        config = _integration_config()
        for context, message in (
            ({"enabled": "yes"}, "context.enabled"),
            ({"recent_turns": 0}, "context.recent_turns"),
            ({"max_recent_chars": True}, "context.max_recent_chars"),
        ):
            candidate = dict(config)
            candidate["context"] = context
            with self.assertRaisesRegex(ValueError, message):
                _validate_runtime_config(candidate)

    def test_state_summary_exposes_authoritative_pending_budget_and_recovery_facts(self):
        """Derive current facts directly from state instead of conversation history."""
        config = _integration_config()
        state = _initial_state("summary", "improve score", config)
        state["plan"] = {"steps": [
            {"id": "inspect", "status": "completed", "description": "Inspect", "next_action": ""},
            {"id": "run", "status": "in_progress", "description": "Run candidate", "next_action": "Evaluate"},
        ]}
        state["phase"] = "experiment"
        state["iteration"] = 4
        state["tool_call_count"] = 21
        state["llm_call_count"] = 22
        state["current_config"] = {"learning_rate": 0.6}
        state["best_config"] = {"learning_rate": 0.6}
        state["best_metrics"] = {"status": "success", "score": 10.96}
        state["best_source"] = "iteration_02"
        state["pending_run"] = {
            "iteration": 4, "hypothesis": "try exploration", "expected_effect": "higher score",
            "changes": {"epsilon": 0.01}, "config_candidate": {"learning_rate": 0.6, "epsilon": 0.01},
            "metrics": {"status": "success", "score": 6.71}, "verification_errors": [],
            "snapshot_id": "iteration_04_before.yaml", "evaluation": {
                "llm_decision": "rollback", "harness_decision": "rollback",
            },
        }
        summary = _state_summary(state)
        self.assertEqual(summary["run"]["phase"], "experiment")
        self.assertEqual(summary["progress"]["iteration"], 4)
        self.assertEqual(summary["best"]["metrics"]["score"], 10.96)
        self.assertEqual(summary["pending_run"]["snapshot_id"], "iteration_04_before.yaml")
        self.assertEqual(summary["last_result"]["decision"], "rollback")
        self.assertFalse(summary["last_result"]["restore_verified"])
        self.assertEqual(summary["remaining_budget"]["candidates"], 1)
        self.assertEqual(summary["remaining_budget"]["tool_calls"], 9)
        self.assertEqual(summary["plan"]["current_step"]["id"], "run")

        state["pending_run"] = None
        state["history"] = [{
            "iteration": 4, "changes": {"epsilon": 0.01},
            "metrics": {"status": "success", "score": 6.71}, "decision": "rollback",
            "recovery": {"snapshot_id": "iteration_04_before.yaml", "sha256": "abc"},
        }]
        restored = _state_summary(state)
        self.assertIsNone(restored["pending_run"])
        self.assertTrue(restored["last_result"]["restore_verified"])

    def test_state_summary_compacts_bulky_last_tool_payloads(self):
        """Preserve the latest tool outcome without copying large content into every prompt."""
        state = _initial_state("last-tool", "goal", _integration_config())
        state["last_tool"] = {
            "name": "read_file", "tool_call": 3, "ok": True,
            "result": {
                "ok": True, "path": "task/source.py", "content": "x" * 12000,
                "stdout_tail": "log", "files": [{"path": "a"}, {"path": "b"}],
            },
        }
        state["last_tool"]["result"] = _compact_tool_result(state["last_tool"]["result"])
        summary = _state_summary(state)
        result = summary["last_tool"]["result"]
        self.assertEqual(result["path"], "task/source.py")
        self.assertNotIn("content", result)
        self.assertEqual(result["omitted_payloads"]["content"]["characters"], 12000)
        self.assertEqual(result["omitted_payloads"]["files"]["items"], 2)

    def test_context_window_keeps_complete_recent_turns_and_authoritative_summary(self):
        """Compact history without orphaning tool results or dropping Kimi reasoning fields."""
        state = _initial_state("window", "goal", _integration_config())
        messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "goal"}]
        for index in range(1, 6):
            messages.extend([
                {
                    "role": "assistant", "content": "", "reasoning_content": f"reason-{index}",
                    "tool_calls": [{
                        "id": f"call_{index}", "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }],
                },
                {"role": "tool", "tool_call_id": f"call_{index}", "name": "read_file", "content": "ok"},
            ])
        config = _integration_config()
        config["context"] = {"enabled": True, "recent_turns": 4, "max_recent_chars": 24000}
        compact, record = _build_model_context(messages, state, config)
        self.assertEqual(record["total_turn_groups"], 5)
        self.assertEqual(record["retained_turn_groups"], 4)
        self.assertEqual(record["omitted_turn_groups"], 1)
        self.assertNotIn("reason-1", json.dumps(compact))
        self.assertIn("reason-5", json.dumps(compact))
        assistant_ids = {
            call["id"] for message in compact if message.get("role") == "assistant"
            for call in message.get("tool_calls") or []
        }
        tool_ids = {
            message["tool_call_id"] for message in compact if message.get("role") == "tool"
        }
        self.assertEqual(assistant_ids, tool_ids)
        self.assertEqual(compact[-1]["role"], "user")
        self.assertIn("AUTHORITATIVE_STATE_SUMMARY", compact[-1]["content"])
        injected_summary = json.loads(compact[-1]["content"].split("\n", 2)[-1])
        self.assertEqual(injected_summary, record["state_summary"])
        self.assertNotIn(compact[-1], messages)

        config["context"]["max_recent_chars"] = 1
        minimal, minimal_record = _build_model_context(messages, state, config)
        self.assertEqual([message["role"] for message in minimal], ["system", "user", "user"])
        self.assertEqual(minimal_record["retained_turn_groups"], 0)

        config["context"]["enabled"] = False
        full, full_record = _build_model_context(messages, state, config)
        self.assertEqual(full, messages)
        self.assertFalse(full_record["enabled"])

    def test_execution_verification_and_comparison_are_independent(self):
        """Ensure deterministic checks can reject misleading experiment output."""
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
        """Reject reports that omit required sections and iteration evidence."""
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
        """Enforce run-scoped writes and completion requirements."""
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
    """Exercise the complete LLM-to-tool state machine with real subprocesses."""

    def test_llm_plans_and_uses_real_file_and_shell_tools_for_three_iterations(self):
        """Run baseline plus three candidates, including verified rollback."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "task").mkdir()
            (root / "runs").mkdir()
            (root / "task" / "goal.md").write_text("Improve score with real experiments.", encoding="utf-8")
            # The temporary experiment is intentionally small but executes as a
            # real child process and writes the same artifact shape as a task.
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
            # Scripted decisions isolate harness behavior from network/model
            # variability while preserving the native function-calling flow.
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
                ("write_file", {"path": f"{run_prefix}/report.md", "content": VALID_REPORT}),
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
            full_messages = [
                json.loads(line) for line in
                (root / "runs" / run_id / "messages.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            final_llm_dir = root / "runs" / run_id / "llm_calls" / "call_021"
            request = json.loads((final_llm_dir / "request.json").read_text(encoding="utf-8"))
            context = json.loads((final_llm_dir / "context.json").read_text(encoding="utf-8"))
            self.assertEqual(len(full_messages), 44)
            self.assertLess(len(request["messages"]), context["full_message_count"])
            self.assertEqual(len(request["messages"]), context["request_message_count"])
            self.assertEqual(context["retained_turn_groups"], 4)
            self.assertGreater(context["omitted_turn_groups"], 0)
            self.assertIn("AUTHORITATIVE_STATE_SUMMARY", request["messages"][-1]["content"])
            self.assertEqual(context["state_summary"]["progress"]["iteration"], 3)
            self.assertEqual(context["state_summary"]["best"]["metrics"]["score"], 7.0)
            self.assertEqual(context["state_summary"]["last_tool"]["name"], "update_plan")
            self.assertFalse(any(
                message.get("role") == "user"
                and message.get("content", "").startswith("AUTHORITATIVE_STATE_SUMMARY")
                for message in full_messages
            ))
            progress = progress_stream.getvalue()
            for event in ("LLM", "TOOL", "FILE", "EXP", "CHECK", "RECOVER", "DONE"):
                self.assertIn(event, progress)


class ScriptedLLM:
    """Deterministic chat substitute that emits a predefined tool-call trace."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0
        self.reasoning_was_replayed = False

    def __call__(self, messages, _tools, _config, tool_choice):
        """Return the next tool call and verify prior reasoning was replayed."""
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


class FakeOpenAIClientFactory:
    """Capture OpenAI SDK client and completion arguments without networking."""

    def __init__(self, payload):
        self.payload = payload
        self.client_kwargs = None
        self.create_kwargs = None

    def __call__(self, **kwargs):
        """Build the minimal nested client surface consumed by planner.chat."""
        self.client_kwargs = kwargs
        factory = self

        class Completions:
            def create(self, **create_kwargs):
                factory.create_kwargs = create_kwargs

                class Completion:
                    def model_dump(self, mode):
                        if mode != "json":
                            raise AssertionError("Expected JSON serialization")
                        return factory.payload

                return Completion()

        class Chat:
            completions = Completions()

        class Client:
            chat = Chat()

        return Client()


def _integration_config():
    """Return a complete, bounded runtime configuration for integration tests."""
    return {
        "goal_file": "task/goal.md",
        "planner": {"provider": "openai_compatible", "model": "test-model"},
        "context": {"enabled": True, "recent_turns": 4, "max_recent_chars": 24000},
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
    """Build a valid run_command payload for the temporary experiment."""
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
    """Build a completed update_plan payload with supporting evidence."""
    return {"step_id": step_id, "status": "completed", "evidence": evidence, "next_action": next_action}
