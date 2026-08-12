import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

from .execution import (
    activate_config,
    compare_metrics,
    fingerprint,
    load_yaml,
    run_experiment,
    target_reached,
    validate_proposal,
    write_json,
    write_yaml,
)
from .planner import PlannerError, propose
from .report import generate_report


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def run_agent(config_path, run_id=None):
    config_path = _root_path(config_path)
    config = load_yaml(config_path)
    _validate_runtime_config(config)
    run_id = run_id or datetime.now().strftime("autoresearch_%Y%m%d_%H%M%S")
    if not run_id.strip() or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one directory name without path separators")
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    goal = _root_path(config["goal_file"]).read_text(encoding="utf-8")
    state = _initial_state(run_id, goal, config)
    current_path = run_dir / "current_config.yaml"
    trajectory_path = run_dir / "trajectory.jsonl"
    write_yaml(run_dir / "run_config.yaml", config)
    write_yaml(current_path, config["baseline"])
    _save_state(run_dir, state)

    try:
        baseline_dir = run_dir / "baseline"
        baseline_dir.mkdir()
        shutil.copy2(current_path, baseline_dir / "config.yaml")
        execution, metrics = run_experiment(
            ROOT,
            config,
            current_path,
            f"{run_id}/baseline/raw",
            baseline_dir,
            _experiment_timeout(state),
        )
        baseline_errors = execution["verification_errors"]
        state["baseline"] = {
            "status": "failed" if baseline_errors else "success",
            "config": load_yaml(current_path),
            "metrics": metrics,
            "execution": execution,
        }
        _record(trajectory_path, {
            "event": "baseline",
            "iteration": 0,
            "execution": execution,
            "metrics": metrics,
            "errors": baseline_errors,
            "next_action": "stop" if baseline_errors else "plan_next_experiment",
        })
        if baseline_errors:
            state["stop_reason"] = "unrecoverable_error"
            state["status"] = "failed"
            state["consecutive_failures"] = 1
            state["total_failures"] = 1
            state["failures"].append({
                "agent_step": 0,
                "type": "baseline_failure",
                "message": "; ".join(baseline_errors),
                "recovery_action": "none; a valid baseline is required",
            })
        else:
            state["current_config"] = copy.deepcopy(config["baseline"])
            state["best_config"] = copy.deepcopy(config["baseline"])
            state["best_metrics"] = metrics
            state["best_source"] = "baseline"
        _save_state(run_dir, state)

        tested = {fingerprint(config["baseline"])}
        while not state["stop_reason"]:
            stop_reason = _stop_reason(state, config)
            if stop_reason:
                state["stop_reason"] = stop_reason
                break

            state["agent_steps"] += 1
            step = state["agent_steps"]
            context = _planner_context(state, config)
            try:
                proposal, prompt, raw = propose(context, config["planner"])
            except (PlannerError, OSError, ValueError) as exc:
                _planning_failure(state, run_dir, trajectory_path, step, exc)
                _save_state(run_dir, state)
                continue

            attempt_dir = run_dir / "planning_attempts" / f"step_{step:02d}"
            attempt_dir.mkdir(parents=True)
            (attempt_dir / "input.json").write_text(prompt, encoding="utf-8")
            (attempt_dir / "raw_output.txt").write_text(raw, encoding="utf-8")
            write_json(attempt_dir / "proposal.json", proposal)

            errors, candidate_config = validate_proposal(
                proposal, state["current_config"], config["search_space"], tested
            )
            if errors:
                _proposal_failure(state, trajectory_path, step, proposal, errors)
                write_json(attempt_dir / "validation.json", {"ok": False, "errors": errors})
                _save_state(run_dir, state)
                continue

            write_json(attempt_dir / "validation.json", {"ok": True, "errors": []})
            state["iteration"] += 1
            iteration = state["iteration"]
            iteration_dir = run_dir / f"iteration_{iteration:02d}"
            iteration_dir.mkdir()
            shutil.copy2(current_path, iteration_dir / "before_config.yaml")
            write_yaml(iteration_dir / "after_config.yaml", candidate_config)
            write_json(iteration_dir / "proposal.json", proposal)
            activate_config(iteration_dir / "after_config.yaml", current_path)
            tested.add(fingerprint(candidate_config))

            execution, metrics = run_experiment(
                ROOT,
                config,
                current_path,
                f"{run_id}/iteration_{iteration:02d}/raw",
                iteration_dir,
                _experiment_timeout(state),
            )
            verification_errors = execution["verification_errors"]
            if verification_errors:
                decision = "rollback"
                reason = "; ".join(verification_errors)
                recovery = _rollback(iteration_dir, current_path, state["best_config"])
                state["consecutive_failures"] += 1
                state["total_failures"] += 1
                state["failures"].append({
                    "agent_step": step,
                    "iteration": iteration,
                    "type": "execution_failure",
                    "message": reason,
                    "recovery_action": recovery,
                })
            else:
                try:
                    decision, reason = compare_metrics(metrics, state["best_metrics"], config["objective"])
                except (KeyError, TypeError, ValueError) as exc:
                    decision = "rollback"
                    reason = f"metric comparison failed: {exc}"
                    recovery = _rollback(iteration_dir, current_path, state["best_config"])
                    state["consecutive_failures"] += 1
                    state["total_failures"] += 1
                    state["failures"].append({
                        "agent_step": step,
                        "iteration": iteration,
                        "type": "metric_failure",
                        "message": reason,
                        "recovery_action": recovery,
                    })
                else:
                    state["consecutive_failures"] = 0
                    if decision == "accept":
                        recovery = None
                        state["current_config"] = copy.deepcopy(candidate_config)
                        state["best_config"] = copy.deepcopy(candidate_config)
                        state["best_metrics"] = metrics
                        state["best_source"] = f"iteration_{iteration:02d}"
                    else:
                        recovery = _rollback(iteration_dir, current_path, state["best_config"])

            item = {
                "iteration": iteration,
                "agent_step": step,
                "proposal": proposal,
                "config_before": load_yaml(iteration_dir / "before_config.yaml"),
                "config_candidate": candidate_config,
                "execution_status": "failed" if verification_errors else "success",
                "execution": execution,
                "metrics": metrics,
                "decision": decision,
                "reason": reason,
                "recovery_action": recovery,
                "next_action": "check_stop_conditions",
            }
            state["history"].append(item)
            _record(trajectory_path, {"event": "iteration", **item})
            _save_state(run_dir, state)

        if not state["stop_reason"]:
            state["stop_reason"] = "unrecoverable_error"
        if state["status"] == "running":
            state["status"] = "completed"
    except Exception as exc:
        state["status"] = "failed"
        state["stop_reason"] = "unrecoverable_error"
        state["failures"].append({
            "agent_step": state["agent_steps"],
            "type": type(exc).__name__,
            "message": str(exc),
            "recovery_action": "none",
            "traceback": traceback.format_exc(),
        })
        _record(trajectory_path, {
            "event": "unrecoverable_error",
            "error": state["failures"][-1],
            "next_action": "stop",
        })
    finally:
        state["finished_at"] = _now()
        state["duration_seconds"] = time.time() - state["started_unix"]
        state.pop("started_unix", None)
        _record(trajectory_path, {
            "event": "stop",
            "iteration": state["iteration"],
            "agent_steps": state["agent_steps"],
            "stop_reason": state["stop_reason"],
            "best_source": state["best_source"],
            "best_metrics": state["best_metrics"],
        })
        _save_state(run_dir, state)
        generate_report(state, run_dir / "report.md")

    return {
        "status": state["status"],
        "stop_reason": state["stop_reason"],
        "run_dir": str(run_dir),
        "state_path": str(run_dir / "state.json"),
        "report_path": str(run_dir / "report.md"),
    }


def _initial_state(run_id, goal, config):
    budget = config["budget"]
    return {
        "run_id": run_id,
        "status": "running",
        "goal": goal,
        "objective": copy.deepcopy(config["objective"]),
        "started_at": _now(),
        "started_unix": time.time(),
        "finished_at": None,
        "duration_seconds": None,
        "iteration": 0,
        "agent_steps": 0,
        "consecutive_failures": 0,
        "total_failures": 0,
        "limits": {
            "max_iterations": int(budget["max_iterations"]),
            "max_agent_steps": int(budget["max_agent_steps"]),
            "max_consecutive_failures": int(budget["max_consecutive_failures"]),
            "max_wall_time_minutes": float(budget["max_wall_time_minutes"]),
            "experiment_timeout_seconds": float(budget["experiment_timeout_seconds"]),
        },
        "baseline": None,
        "current_config": None,
        "best_config": None,
        "best_metrics": None,
        "best_source": None,
        "history": [],
        "failures": [],
        "stop_reason": None,
    }


def _planner_context(state, config):
    return {
        "research_goal": state["goal"],
        "objective": state["objective"],
        "current_config": state["current_config"],
        "best_metrics": state["best_metrics"],
        "previous_experiments": [
            {
                "iteration": item["iteration"],
                "hypothesis": item["proposal"]["hypothesis"],
                "changes": item["proposal"]["changes"],
                "metrics": item["metrics"],
                "decision": item["decision"],
                "reason": item["reason"],
            }
            for item in state["history"]
        ],
        "previous_failures": state["failures"],
        "remaining_budget": {
            "experiments": state["limits"]["max_iterations"] - state["iteration"],
            "agent_steps": state["limits"]["max_agent_steps"] - state["agent_steps"],
            "recoverable_failures": (
                state["limits"]["max_consecutive_failures"] - state["consecutive_failures"]
            ),
        },
        "allowed_changes": config["search_space"],
    }


def _planning_failure(state, run_dir, trajectory_path, step, exc):
    failure_dir = run_dir / "planning_attempts" / f"step_{step:02d}"
    failure_dir.mkdir(parents=True, exist_ok=True)
    failure = {
        "agent_step": step,
        "type": "planning_failure",
        "message": str(exc),
        "recovery_action": "retry planning with failure included in context",
    }
    write_json(failure_dir / "error.json", failure)
    state["failures"].append(failure)
    state["consecutive_failures"] += 1
    state["total_failures"] += 1
    _record(trajectory_path, {
        "event": "planning_failure",
        **failure,
        "next_action": "retry_or_stop_at_failure_limit",
    })


def _proposal_failure(state, trajectory_path, step, proposal, errors):
    failure = {
        "agent_step": step,
        "type": "invalid_proposal",
        "message": "; ".join(errors),
        "recovery_action": "reject without execution and re-plan",
    }
    state["failures"].append(failure)
    state["consecutive_failures"] += 1
    state["total_failures"] += 1
    _record(trajectory_path, {
        "event": "invalid_proposal",
        "proposal": proposal,
        **failure,
        "next_action": "retry_or_stop_at_failure_limit",
    })


def _rollback(iteration_dir, current_path, best_config):
    restored = activate_config(iteration_dir / "before_config.yaml", current_path)
    if restored != best_config:
        raise RuntimeError("Rollback restored a config that differs from canonical best_config")
    return "restored before_config.yaml and verified it equals best_config"


def _stop_reason(state, config):
    if state["iteration"] >= state["limits"]["max_iterations"]:
        return "iteration_budget_exhausted"
    if state["agent_steps"] >= state["limits"]["max_agent_steps"]:
        return "agent_step_budget_exhausted"
    if state["consecutive_failures"] >= state["limits"]["max_consecutive_failures"]:
        return "too_many_failures"
    elapsed_minutes = (time.time() - state["started_unix"]) / 60
    if elapsed_minutes >= state["limits"]["max_wall_time_minutes"]:
        return "wall_time_budget_exhausted"
    if target_reached(state["best_metrics"], config["objective"]):
        return "target_reached"
    return None


def _experiment_timeout(state):
    wall_deadline = state["started_unix"] + state["limits"]["max_wall_time_minutes"] * 60
    remaining_wall_time = max(0.1, wall_deadline - time.time())
    return min(state["limits"]["experiment_timeout_seconds"], remaining_wall_time)


def _validate_runtime_config(config):
    required = ("goal_file", "planner", "budget", "experiment", "objective", "baseline", "search_space")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Runtime config is missing sections: {', '.join(missing)}")
    budget_keys = (
        "max_iterations", "max_agent_steps", "max_consecutive_failures",
        "max_wall_time_minutes", "experiment_timeout_seconds",
    )
    for key in budget_keys:
        if float(config["budget"].get(key, 0)) <= 0:
            raise ValueError(f"budget.{key} must be positive")
    if config["objective"].get("direction", "maximize") not in {"maximize", "minimize"}:
        raise ValueError("objective.direction must be maximize or minimize")
    if not config["objective"].get("metric"):
        raise ValueError("objective.metric is required")
    if not isinstance(config["experiment"].get("command"), list) or not config["experiment"]["command"]:
        raise ValueError("experiment.command must be a non-empty list")
    if not config["experiment"].get("metrics_path"):
        raise ValueError("experiment.metrics_path is required")
    if not isinstance(config["search_space"], dict) or not config["search_space"]:
        raise ValueError("search_space must be a non-empty object")
    provider = config["planner"].get("provider", "openai_compatible")
    if provider == "openai_compatible" and not config["planner"].get("model"):
        raise ValueError("planner.model is required for openai_compatible")
    if provider == "local_command" and not config["planner"].get("command"):
        raise ValueError("planner.command is required for local_command")
    if provider not in {"openai_compatible", "local_command"}:
        raise ValueError(f"Unsupported planner provider: {provider}")


def _save_state(run_dir, state):
    state["remaining_budget"] = {
        "experiments": max(0, state["limits"]["max_iterations"] - state["iteration"]),
        "agent_steps": max(0, state["limits"]["max_agent_steps"] - state["agent_steps"]),
        "consecutive_failures": max(
            0,
            state["limits"]["max_consecutive_failures"] - state["consecutive_failures"],
        ),
    }
    write_json(run_dir / "state.json", state)


def _record(path, event):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": _now(), **event}) + "\n")


def _root_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _now():
    return datetime.now(timezone.utc).isoformat()
