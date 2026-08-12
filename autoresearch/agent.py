"""Core LLM-driven control loop for iterative machine-learning experiments."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

from .execution import load_yaml, write_json, write_yaml
from .planner import PlannerError, chat, parse_tool_call
from .progress import ProgressReporter, tool_summary
from .tooling import ToolHarness, tool_schemas


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def run_agent(
    config_path, run_id=None, goal=None, *, chat_fn=None, root=None, runs_root=None,
    reporter=None, progress_mode="normal",
):
    """Run the function-calling research agent until finish or a safety limit.

    Every LLM request, assistant response, tool call, state transition, and
    terminal outcome is persisted under a unique run directory for replay and
    auditability.
    """
    root = Path(root or ROOT).resolve()
    runs_root = Path(runs_root or (root / "runs")).resolve()
    config_path = _root_path(config_path, root)
    config = load_yaml(config_path)
    _validate_runtime_config(config)
    run_id = run_id or datetime.now().strftime("autoresearch_%Y%m%d_%H%M%S")
    _validate_run_id(run_id)
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    if goal is None:
        goal = _root_path(config["goal_file"], root).read_text(encoding="utf-8").strip()
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("A non-empty natural-language goal is required")
    goal = goal.strip()
    reporter = reporter or ProgressReporter(progress_mode)
    reporter.emit("RUN", f"started {run_id}; artifacts: {run_dir}")

    state = _initial_state(run_id, goal, config)
    trajectory_path = run_dir / "trajectory.jsonl"
    messages_path = run_dir / "messages.jsonl"
    write_yaml(run_dir / "run_config.yaml", config)
    _save_state(run_dir, state)

    schemas = tool_schemas()
    messages = [
        {"role": "system", "content": _system_prompt(root, run_dir, config, state)},
        {"role": "user", "content": goal},
    ]
    for message in messages:
        _append_jsonl(messages_path, message)
    harness = ToolHarness(root, run_dir, config, state, reporter=reporter)
    if chat_fn is None:
        def llm(messages_arg, schemas_arg, planner_config, tool_choice):
            return chat(
                messages_arg, schemas_arg, planner_config, tool_choice, reporter=reporter
            )
    else:
        llm = chat_fn

    try:
        while state["status"] == "running":
            safety_reason = _safety_stop_reason(state)
            if safety_reason:
                state["status"] = "failed" if safety_reason == "too_many_failures" else "incomplete_requirements"
                state["stop_reason"] = safety_reason
                reporter.emit("STOP", f"safety limit reached: {safety_reason}")
                break

            state["llm_call_count"] += 1
            state["agent_steps"] = state["llm_call_count"]
            llm_dir = run_dir / "llm_calls" / f"call_{state['llm_call_count']:03d}"
            llm_dir.mkdir(parents=True)
            # Persist the exact model context before the network call so a
            # stalled or failed request remains diagnosable after termination.
            request_body = {
                "model": config["planner"]["model"], "messages": messages,
                "tools": schemas, "tool_choice": "required", "parallel_tool_calls": False,
            }
            write_json(llm_dir / "request.json", request_body)
            llm_started = time.monotonic()
            reporter.emit("LLM", f"call {state['llm_call_count']} started")
            try:
                with reporter.waiting("LLM", f"call {state['llm_call_count']} waiting"):
                    assistant, raw_response = llm(
                        messages, schemas, config["planner"], "required"
                    )
            except (PlannerError, OSError, ValueError) as exc:
                _record_llm_failure(state, trajectory_path, exc)
                write_json(llm_dir / "error.json", {"type": type(exc).__name__, "message": str(exc)})
                reporter.emit("ERROR", f"LLM call {state['llm_call_count']}: {exc}")
                _save_state(run_dir, state)
                continue

            llm_duration = time.monotonic() - llm_started
            usage = raw_response.get("usage") if isinstance(raw_response, dict) else None
            tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
            token_text = f", tokens={tokens}" if tokens is not None else ""
            reporter.emit(
                "LLM", f"call {state['llm_call_count']} completed in {llm_duration:.1f}s{token_text}"
            )

            write_json(llm_dir / "response.json", raw_response)
            messages.append(assistant)
            _append_jsonl(messages_path, assistant)
            calls = assistant.get("tool_calls") or []
            if len(calls) != 1:
                state["consecutive_failures"] += 1
                state["total_failures"] += 1
                failure = {
                    "type": "invalid_llm_action", "llm_call": state["llm_call_count"],
                    "message": f"Expected exactly one tool call, received {len(calls)}",
                    "recovery_action": "ask the LLM for exactly one tool call",
                }
                state["failures"].append(failure)
                _record(trajectory_path, {"event": "invalid_llm_action", **failure})
                reporter.emit("ERROR", failure["message"])
                if calls:
                    for raw_call in calls:
                        call_id = str(raw_call.get("id") or f"invalid_{state['llm_call_count']}")
                        tool_message = {
                            "role": "tool", "tool_call_id": call_id,
                            "name": str((raw_call.get("function") or {}).get("name") or "invalid"),
                            "content": json.dumps({"ok": False, "error": failure["message"]}),
                        }
                        messages.append(tool_message)
                        _append_jsonl(messages_path, tool_message)
                else:
                    correction = {"role": "user", "content": failure["message"] + ". Call one available tool now."}
                    messages.append(correction)
                    _append_jsonl(messages_path, correction)
                _save_state(run_dir, state)
                continue

            raw_call = calls[0]
            try:
                call_id, tool_name, arguments = parse_tool_call(raw_call)
            except PlannerError as exc:
                state["consecutive_failures"] += 1
                state["total_failures"] += 1
                failure = {
                    "type": "invalid_tool_call", "llm_call": state["llm_call_count"],
                    "message": str(exc), "recovery_action": "ask the LLM to correct tool arguments",
                }
                state["failures"].append(failure)
                _record(trajectory_path, {"event": "invalid_tool_call", **failure})
                reporter.emit("ERROR", f"invalid tool call: {exc}")
                call_id = str(raw_call.get("id") or f"invalid_{state['llm_call_count']}")
                tool_name = str((raw_call.get("function") or {}).get("name") or "invalid")
                result = {"ok": False, "error": str(exc)}
            else:
                if state["tool_call_count"] >= state["limits"]["max_tool_calls"]:
                    state["status"] = "incomplete_requirements"
                    state["stop_reason"] = "tool_call_budget_exhausted"
                    break
                state["tool_call_count"] += 1
                call_dir = run_dir / "tool_calls" / f"call_{state['tool_call_count']:03d}_{tool_name}"
                call_dir.mkdir(parents=True)
                write_json(call_dir / "arguments.json", arguments)
                summary = tool_summary(tool_name, arguments)
                suffix = f" {summary}" if summary else ""
                reporter.emit("TOOL", f"call {state['tool_call_count']} {tool_name}{suffix}")
                result = harness.execute(tool_name, arguments, call_dir)
                write_json(call_dir / "result.json", result)
                if result.get("ok"):
                    reporter.emit("TOOL", f"call {state['tool_call_count']} {tool_name} completed", verbose=True)
                _record(trajectory_path, {
                    "event": "tool_call", "llm_call": state["llm_call_count"],
                    "tool_call": state["tool_call_count"], "tool_call_id": call_id,
                    "tool": tool_name, "arguments": arguments, "result": result,
                    "next_action": result.get("next_action"),
                })

            tool_message = {
                "role": "tool", "tool_call_id": call_id, "name": tool_name,
                "content": json.dumps(result, ensure_ascii=False),
            }
            messages.append(tool_message)
            _append_jsonl(messages_path, tool_message)
            _refresh_requirements(state, config)
            _save_state(run_dir, state)
    except Exception as exc:
        state["status"] = "failed"
        state["stop_reason"] = "unrecoverable_error"
        failure = {
            "type": type(exc).__name__, "message": str(exc),
            "recovery_action": "none", "traceback": traceback.format_exc(),
        }
        state["failures"].append(failure)
        state["total_failures"] += 1
        _record(trajectory_path, {"event": "unrecoverable_error", "error": failure})
        reporter.emit("ERROR", f"unrecoverable: {type(exc).__name__}: {exc}")
    finally:
        state["finished_at"] = _now()
        state["duration_seconds"] = time.time() - state["started_unix"]
        _refresh_requirements(state, config)
        _record(trajectory_path, {
            "event": "stop", "status": state["status"], "stop_reason": state["stop_reason"],
            "iterations": state["iteration"], "llm_calls": state["llm_call_count"],
            "tool_calls": state["tool_call_count"], "best_source": state["best_source"],
            "best_metrics": state["best_metrics"], "requirements": state["requirements"],
        })
        _save_state(run_dir, state)
        reporter.emit(
            "STOP",
            f"status={state['status']}, reason={state['stop_reason']}, "
            f"iterations={state['iteration']}, tool_calls={state['tool_call_count']}",
        )

    report_path = run_dir / "report.md"
    return {
        "status": state["status"], "stop_reason": state["stop_reason"],
        "run_dir": str(run_dir), "state_path": str(run_dir / "state.json"),
        "report_path": str(report_path) if report_path.exists() else None,
    }


def _initial_state(run_id, goal, config):
    """Create the serializable state machine for a new research run."""
    budget = config["budget"]
    return {
        "run_id": run_id, "status": "running", "phase": "planning", "goal": goal,
        "objective": copy.deepcopy(config["objective"]), "started_at": _now(),
        "started_unix": time.time(), "finished_at": None, "duration_seconds": None,
        "iteration": 0, "agent_steps": 0, "llm_call_count": 0, "tool_call_count": 0,
        "consecutive_failures": 0, "total_failures": 0,
        "limits": {
            "min_iterations": int(budget["min_iterations"]),
            "max_iterations": int(budget["max_iterations"]),
            "max_tool_calls": int(budget["max_tool_calls"]),
            "max_consecutive_failures": int(budget["max_consecutive_failures"]),
            "max_wall_time_minutes": float(budget["max_wall_time_minutes"]),
            "experiment_timeout_seconds": float(budget["experiment_timeout_seconds"]),
        },
        "plan": None, "baseline": None, "current_config": None,
        "best_config": None, "best_metrics": None, "best_source": None,
        "tested_fingerprints": [], "pending_run": None, "history": [],
        "failures": [], "recovery_events": [], "requirements": {
            "min_iterations_met": False, "recovery_demonstrated": False, "report_verified": False,
        },
        "stop_reason": None, "agent_summary": None,
    }


def _system_prompt(root, run_dir, config, state):
    """Build the policy and experiment protocol supplied to the planner."""
    relative_run = run_dir.relative_to(root).as_posix()
    script = config["experiment"]["command"][1]
    timeout = state["limits"]["experiment_timeout_seconds"]
    return f"""You are an autonomous machine-learning research Agent operating through real tools.

You, not the harness, must plan the work, inspect files, write experiment configuration, run experiments, inspect metrics/logs, evaluate results, request rollback when needed, update the plan, write the final report, and call finish.

Research constraints:
- Run directory: {relative_run}
- Runtime configuration: {relative_run}/run_config.yaml
- Natural-language goal file: {config['goal_file']}
- Objective: {json.dumps(config['objective'])}
- Candidate experiments required: at least {state['limits']['min_iterations']}, at most {state['limits']['max_iterations']}; baseline does not count.
- A measured candidate regression or failed experiment followed by verified snapshot rollback must be demonstrated before finish. Do not intentionally corrupt a command or configuration to manufacture failure.
- Only write inside {relative_run}; source code and data are read-only.
- Start by calling submit_plan with at least five executable steps. Do not provide a final answer instead of using tools.

Experiment protocol:
1. Inspect the goal, runtime config, task code/data as useful.
2. Write the exact configured baseline YAML to {relative_run}/current_config.yaml.
3. Run baseline with argv ["python", "{script}", "--config", "{relative_run}/current_config.yaml", "--run-id", "{state['run_id']}/baseline/raw", "--timeout-seconds", "{timeout}"].
4. For each candidate, modify current_config.yaml only with allowed search-space values, then run the same script with run-id {state['run_id']}/iteration_NN/raw.
5. Inspect each returned metric/log, call evaluate_result, and call restore_snapshot if rollback is required.
6. Write {relative_run}/report.md with these exact English section headings: Research Goal, Plan, Baseline, Experiment Process, Failure and Recovery, Best Result, Limitations. Document every iteration as 'Iteration N' and include the exact best metric value.
7. Call finish. If a tool rejects an action, reflect on its error and correct it.

Use one tool call per assistant turn. Never claim a tool ran unless its tool response says ok=true."""


def _safety_stop_reason(state):
    """Return the first exhausted safety budget, if any."""
    if state["tool_call_count"] >= state["limits"]["max_tool_calls"]:
        return "tool_call_budget_exhausted"
    if state["consecutive_failures"] >= state["limits"]["max_consecutive_failures"]:
        return "too_many_failures"
    elapsed_minutes = (time.time() - state["started_unix"]) / 60
    if elapsed_minutes >= state["limits"]["max_wall_time_minutes"]:
        return "wall_time_budget_exhausted"
    return None


def _validate_runtime_config(config):
    """Validate required sections and enforce minimum harness guarantees."""
    required = ("goal_file", "planner", "budget", "experiment", "objective", "baseline", "search_space", "tools")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Runtime config is missing sections: {', '.join(missing)}")
    budget_keys = (
        "min_iterations", "max_iterations", "max_tool_calls", "max_consecutive_failures",
        "max_wall_time_minutes", "experiment_timeout_seconds",
    )
    for key in budget_keys:
        if float(config["budget"].get(key, 0)) <= 0:
            raise ValueError(f"budget.{key} must be positive")
    if int(config["budget"]["min_iterations"]) < 3:
        raise ValueError("budget.min_iterations must be at least 3")
    if int(config["budget"]["max_iterations"]) < int(config["budget"]["min_iterations"]):
        raise ValueError("budget.max_iterations must be >= min_iterations")
    if config["planner"].get("provider", "openai_compatible") != "openai_compatible":
        raise ValueError("planner.provider must be openai_compatible")
    if not config["planner"].get("model"):
        raise ValueError("planner.model is required")
    if config["objective"].get("direction", "maximize") not in {"maximize", "minimize"}:
        raise ValueError("objective.direction must be maximize or minimize")
    if not config["objective"].get("metric"):
        raise ValueError("objective.metric is required")
    command = config["experiment"].get("command")
    if not isinstance(command, list) or len(command) < 2:
        raise ValueError("experiment.command must contain Python and a script")
    if not config["experiment"].get("metrics_path"):
        raise ValueError("experiment.metrics_path is required")
    if not isinstance(config["search_space"], dict) or not config["search_space"]:
        raise ValueError("search_space must be a non-empty object")
    if not isinstance(config["tools"].get("read_roots"), list):
        raise ValueError("tools.read_roots must be a list")
    if not config["tools"].get("write_root"):
        raise ValueError("tools.write_root is required")
    allowed_commands = config["tools"].get("allowed_commands")
    if not isinstance(allowed_commands, list) or not allowed_commands:
        raise ValueError("tools.allowed_commands must be a non-empty list")
    expected_command = f"python {command[1]}"
    if expected_command not in allowed_commands:
        raise ValueError(f"tools.allowed_commands must contain {expected_command!r}")


def _validate_run_id(run_id):
    """Reject run identifiers that could escape the configured runs root."""
    if not isinstance(run_id, str) or not run_id.strip() or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one directory name without path separators")


def _refresh_requirements(state, config):
    """Recompute completion evidence and remaining budgets in persisted state."""
    recovery_required = config.get("evidence", {}).get("require_recovery_event", True)
    recovery_observed = any(
        item.get("qualifies") and item.get("type") == "experiment_rollback"
        for item in state["recovery_events"]
    )
    state["requirements"] = {
        "min_iterations_met": state["iteration"] >= state["limits"]["min_iterations"],
        "recovery_demonstrated": recovery_observed or not recovery_required,
        "report_verified": state["requirements"].get("report_verified", False),
    }
    state["remaining_budget"] = {
        "candidate_experiments": max(0, state["limits"]["max_iterations"] - state["iteration"]),
        "tool_calls": max(0, state["limits"]["max_tool_calls"] - state["tool_call_count"]),
        "consecutive_failures": max(0, state["limits"]["max_consecutive_failures"] - state["consecutive_failures"]),
    }


def _record_llm_failure(state, path, exc):
    """Record a recoverable LLM failure and advance failure counters."""
    state["consecutive_failures"] += 1
    state["total_failures"] += 1
    failure = {
        "type": "llm_call_failure", "llm_call": state["llm_call_count"],
        "message": str(exc), "recovery_action": "retry with the same persisted conversation",
    }
    state["failures"].append(failure)
    _record(path, {"event": "llm_call_failure", **failure})


def _save_state(run_dir, state):
    write_json(Path(run_dir) / "state.json", state)


def _record(path, event):
    _append_jsonl(path, {"timestamp": _now(), **event})


def _append_jsonl(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _root_path(path, root):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def _now():
    return datetime.now(timezone.utc).isoformat()
