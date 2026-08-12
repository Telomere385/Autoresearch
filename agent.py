# 这是 AutoResearch Agent 的主入口。
# 它负责读取任务配置和目标，构造 LLM planner 上下文，执行 baseline 与候选实验，并把决策、状态和报告写入 runs/。
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import yaml

from planner import build_planner
from scripts.vcs import commit_paths


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
TASKS_DIR = BASE_DIR / "tasks"


class ProgressBar:
    def __init__(self, total_steps):
        self.total_steps = max(1, int(total_steps))
        self.current = 0
        self.last_line_length = 0
        self.active_line = False

    def update(self, step, message):
        self.current = min(self.total_steps, max(self.current, int(step)))
        width = 28
        filled = int(width * self.current / self.total_steps)
        bar = "#" * filled + "-" * (width - filled)
        line = f"[{bar}] {self.current}/{self.total_steps} {message}"
        padding = " " * max(0, self.last_line_length - len(line))
        print(f"\r{line}{padding}", end="", flush=True)
        self.last_line_length = len(line)
        self.active_line = True

    def advance(self, message):
        self.update(self.current + 1, message)

    def line(self, message):
        if self.active_line:
            print("", flush=True)
            self.active_line = False
            self.last_line_length = 0
        print(message, flush=True)

    def finish(self, message):
        self.update(self.total_steps, message)
        print("", flush=True)
        self.active_line = False
        self.last_line_length = 0


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_run_dir(task_name, run_id):
    if run_id is None:
        run_id = f"autoresearch_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / task_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def resolve_task(task_name):
    task_dir = TASKS_DIR / task_name
    if not task_dir.exists():
        raise FileNotFoundError(f"Unknown task: {task_name} ({task_dir})")
    return task_dir


def default_agent_config_path(task_dir):
    flat_config = task_dir / "agent.yaml"
    grouped_config = task_dir / "configs" / "agent.yaml"
    if flat_config.exists():
        return flat_config
    return grouped_config


def default_task_manifest_path(task_dir):
    flat_manifest = task_dir / "task.yaml"
    grouped_manifest = task_dir / "manifest" / "task.yaml"
    if flat_manifest.exists():
        return flat_manifest
    return grouped_manifest


def baseline_config_from_agent_config(config):
    budget = config["budget"]
    base = copy.deepcopy(config["base_experiment"])
    base["seed"] = budget.get("seed", base.get("seed", 0))
    base["training_episodes"] = budget["training_episodes"]
    base["evaluation_episodes"] = budget["evaluation_episodes"]
    base["save_every"] = budget.get("save_every", max(1, min(100, budget["training_episodes"])))
    return base


def set_nested(config, key, value):
    parts = key.split(".")
    target = config
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def apply_changes(base_config, changes):
    candidate = copy.deepcopy(base_config)
    for key, value in changes.items():
        set_nested(candidate, key, value)
    return candidate


def config_fingerprint(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def validate_candidate(candidate, search_space, constraints, base_config):
    errors = []
    changes = candidate.get("changes", {})
    if not isinstance(changes, dict):
        errors.append("candidate.changes must be an object")
        return {"valid": False, "errors": errors}
    for key, value in changes.items():
        if key not in search_space:
            errors.append(f"{key} is not in search_space")
            continue
        if value not in search_space[key]:
            errors.append(f"{key}={value!r} is not an allowed value")
    if constraints.get("forbid_state_space_expansion"):
        for forbidden in ("state_discretization.dx_dy_bin_size", "state_discretization.velocity_bin_size"):
            if forbidden in changes:
                errors.append(f"{forbidden} cannot be changed when forbid_state_space_expansion is true")
    if constraints.get("forbid_source_edits") and candidate.get("source_edits"):
        errors.append("source_edits are forbidden")
    return {"valid": not errors, "errors": errors}


def run_task_experiment(task, config_path, run_id, tool_log_path, progress=None):
    interface = task.get("interface", {})
    runner = BASE_DIR / interface.get("experiment_runner", "")
    if not runner.exists():
        raise FileNotFoundError(f"Task runner not found: {runner}")
    cmd = [
        sys.executable,
        str(runner),
        "--config",
        str(config_path),
        "--run-id",
        run_id,
    ]
    started = time.time()
    process = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    stdout_lines = []
    if process.stdout is not None:
        for line in process.stdout:
            stdout_lines.append(line)
            stripped = line.strip()
            if progress and stripped:
                if stripped.startswith("Round "):
                    progress.line(f"  {stripped}")
                elif stripped.startswith('"status"') or stripped.startswith('"mean_score"'):
                    progress.line(f"  {stripped}")
    returncode = process.wait()
    ended = time.time()
    payload = {
        "command": cmd,
        "returncode": returncode,
        "stdout": "".join(stdout_lines),
        "started_at": started,
        "ended_at": ended,
        "duration_seconds": ended - started,
    }
    append_jsonl(tool_log_path, payload)
    return returncode, RUNS_DIR / run_id / "summary.json"


def load_summary(summary_path):
    if not summary_path.exists():
        return {"status": "error", "error": {"message": f"Missing summary: {summary_path}"}}
    return read_json(summary_path)


def metric_value(summary, metric):
    return float(summary.get(metric, 0.0))


def choose_decision(summary, best, objective):
    if summary.get("status") != "success":
        return "rollback", "experiment failed or produced incomplete results"
    if best is None:
        return "accept", "first valid result becomes current best"
    metric = objective.get("primary_metric", "mean_score")
    min_improvement = float(objective.get("min_improvement", 0.0))
    current = metric_value(summary, metric)
    best_value = metric_value(best["summary"], metric)
    if current > best_value + min_improvement:
        return "accept", f"{metric} improved from {best_value} to {current}"
    tie_breaker = objective.get("tie_breaker_metric")
    if tie_breaker and current >= best_value:
        tie_min_improvement = float(objective.get("tie_breaker_min_improvement", 0.0))
        current_tie = metric_value(summary, tie_breaker)
        best_tie = metric_value(best["summary"], tie_breaker)
        if current_tie > best_tie + tie_min_improvement:
            return "accept", f"{metric} tied at {current}; {tie_breaker} improved from {best_tie} to {current_tie}"
    return "reject", f"{metric} did not improve over best value {best_value}"


def reflect_on_result(summary, best, objective):
    metric = objective.get("primary_metric", "mean_score")
    tie_breaker = objective.get("tie_breaker_metric")
    reflection = {
        "check": "completed",
        "summary_status": summary.get("status"),
        "primary_metric": metric,
        "primary_value": metric_value(summary, metric),
        "next_action": "continue",
    }
    if tie_breaker:
        reflection["tie_breaker_metric"] = tie_breaker
        reflection["tie_breaker_value"] = metric_value(summary, tie_breaker)
    if summary.get("status") != "success":
        reflection["finding"] = "experiment failed or summary is incomplete"
        reflection["next_action"] = "rollback"
        return reflection
    if best:
        best_primary = metric_value(best["summary"], metric)
        reflection["best_primary_value"] = best_primary
        if reflection["primary_value"] <= 0.0 and best_primary <= 0.0:
            reflection["finding"] = "primary score remains zero; training budget or exploration may be insufficient"
            reflection["next_action"] = "increase_budget_or_adjust_exploration"
        elif reflection["primary_value"] > best_primary:
            reflection["finding"] = "primary metric improved"
            reflection["next_action"] = "accept"
        else:
            reflection["finding"] = "primary metric did not improve"
            reflection["next_action"] = "reject_or_try_alternative"
    else:
        reflection["finding"] = "first valid result"
        reflection["next_action"] = "accept"
    return reflection


def optional_text(path):
    return read_text(path) if path.exists() else ""


def optional_yaml(path):
    return load_yaml(path) if path.exists() else {}


def make_context(config, program, task, goal, iteration, current_config, best, last, history):
    return {
        "program": program,
        "task": task,
        "goal": goal,
        "iteration": iteration,
        "objective": config["objective"],
        "search_space": config["search_space"],
        "constraints": config.get("constraints", {}),
        "current_config": current_config,
        "best": best,
        "last": last,
        "history": history,
    }


def run_failure_demo(run_dir, config, base_config):
    candidate = {
        "candidate_id": "failure_demo",
        "planner": "validation_demo",
        "rationale": "Demonstrate constraint validation and recovery.",
        "changes": {"state_discretization.dx_dy_bin_size": 10},
    }
    validation = validate_candidate(candidate, config["search_space"], config.get("constraints", {}), base_config)
    payload = {
        "iteration": 0,
        "candidate": candidate,
        "validation": validation,
        "decision": "rollback",
        "reason": "invalid candidate rejected before running experiment",
        "recovery_action": "restore baseline config",
    }
    append_jsonl(run_dir / "decisions.jsonl", payload)
    append_jsonl(run_dir / "errors.jsonl", payload)
    return payload


def generate_report(run_dir, goal, config, baseline, history, best):
    lines = [
        "# Mini AutoResearch Report",
        "",
        "## Goal",
        goal.strip(),
        "",
        "## Configuration",
        f"- Planner: {config.get('planner', {}).get('type', 'rules')} / {config.get('planner', {}).get('provider', 'disabled')}",
        f"- Max iterations: {config['budget']['max_iterations']}",
        f"- Seed: {config['budget'].get('seed', 0)}",
        f"- Training episodes: {config['budget']['training_episodes']}",
        f"- Evaluation episodes: {config['budget']['evaluation_episodes']}",
        "",
        "## Baseline",
        f"- Status: {baseline['summary'].get('status')}",
        f"- Mean score: {baseline['summary'].get('mean_score')}",
        f"- Mean reward: {metric_value(baseline['summary'], 'mean_reward')}",
        "",
        "## Iterations",
    ]
    for item in history:
        summary = item.get("summary", {})
        lines += [
            f"### Iteration {item['iteration']}",
            f"- Planner: {item['candidate'].get('planner')}",
            f"- Changes: `{json.dumps(item['candidate'].get('changes', {}))}`",
            f"- Status: {summary.get('status')}",
            f"- Mean score: {summary.get('mean_score')}",
            f"- Mean reward: {metric_value(summary, 'mean_reward')}",
            f"- Decision: {item.get('decision')} ({item.get('reason')})",
            "",
        ]
    lines += [
        "## Best Result",
        f"- Source: {best.get('label') if best else 'none'}",
        f"- Mean score: {best['summary'].get('mean_score') if best else 'n/a'}",
        f"- Mean reward: {metric_value(best['summary'], 'mean_reward') if best else 'n/a'}",
        "",
        "## Limitations",
        "- Short smoke runs can keep score at zero; reward is recorded as a diagnostic signal.",
        "- The LLM planner is constrained to proposing YAML parameter changes only.",
        "- Repeated candidate configurations are rejected before running.",
        "- The Pygame environment is headless but still simulates real game frames.",
    ]
    with open(run_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_state(run_dir, state):
    write_json(run_dir / "state.json", state)


def export_best_config(task_dir, best):
    if not best or not best.get("config"):
        return None
    if (task_dir / "configs").exists():
        path = task_dir / "configs" / "best.yaml"
    else:
        path = task_dir / "best_config.yaml"
    write_yaml(path, best["config"])
    return path


def maybe_commit_best_config(config, task_name, task_dir, best):
    vcs_config = config.get("version_control", {})
    if not vcs_config.get("enabled", False):
        return {"status": "skipped", "reason": "version_control.enabled is false"}
    best_path = export_best_config(task_dir, best)
    if not best_path:
        return {"status": "skipped", "reason": "no successful best config"}
    if not vcs_config.get("auto_commit", False) and not vcs_config.get("push", False):
        return {"status": "exported", "path": str(best_path), "reason": "auto_commit is false"}
    message = vcs_config.get("commit_message") or f"Update best AutoResearch config for {task_name}"
    try:
        result = commit_paths(
            [best_path],
            message,
            push=bool(vcs_config.get("push", False)),
            remote=vcs_config.get("remote", "origin"),
            branch=vcs_config.get("branch"),
        )
        result["path"] = str(best_path)
        return result
    except Exception as exc:
        return {"status": "error", "type": type(exc).__name__, "message": str(exc), "path": str(best_path)}


def main():
    parser = argparse.ArgumentParser(description="Run an offline-first, LLM-pluggable Mini AutoResearch loop.")
    parser.add_argument("--task", default="flappy_qlearning")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    started = time.time()
    task_dir = resolve_task(args.task)
    config_path = Path(args.config) if args.config else default_agent_config_path(task_dir)
    config = load_yaml(config_path)
    task = optional_yaml(BASE_DIR / config.get("task_file", str(default_task_manifest_path(task_dir))))
    task_name = args.task
    run_id, run_dir = build_run_dir(task_name, args.run_id)
    max_iterations = int(config["budget"]["max_iterations"])
    progress = ProgressBar(total_steps=2 + max_iterations * 4)
    progress.update(0, "initializing")
    progress.line(
        f"Progress total: baseline=2, each iteration=4, "
        f"iterations={max_iterations}, total={progress.total_steps}"
    )
    program = optional_text(BASE_DIR / config.get("program_file", "docs/PROGRAM.md"))
    goal = read_text(BASE_DIR / config.get("goal_file", str(task_dir / "goal.md")))
    write_yaml(run_dir / "config.yaml", config)
    write_yaml(run_dir / "task.yaml", task)
    with open(run_dir / "program.md", "w", encoding="utf-8") as f:
        f.write(program)
    with open(run_dir / "goal.md", "w", encoding="utf-8") as f:
        f.write(goal)

    state = {
        "status": "running",
        "goal": goal,
        "iteration": 0,
        "max_iterations": config["budget"]["max_iterations"],
        "best": None,
        "history": [],
    }
    write_state(run_dir, state)

    try:
        base_config = baseline_config_from_agent_config(config)
        if config.get("failure_demo", {}).get("enabled", False):
            run_failure_demo(run_dir, config, base_config)
            progress.line("Recorded validation failure demo and rollback.")

        baseline_config_path = run_dir / "baseline_candidate.yaml"
        write_yaml(baseline_config_path, base_config)
        progress.advance("running baseline train/eval")
        baseline_task_run_id = f"{task_name}/{run_id}/baseline"
        _, baseline_summary_path = run_task_experiment(task, baseline_config_path, baseline_task_run_id, run_dir / "tool_calls.jsonl", progress)
        baseline_summary = load_summary(baseline_summary_path)
        baseline = {"label": "baseline", "config": base_config, "summary": baseline_summary}
        best = baseline if baseline_summary.get("status") == "success" else None
        last = baseline
        history = []
        seen_configs = {config_fingerprint(base_config)}
        progress.advance(f"baseline complete mean_score={baseline_summary.get('mean_score')}")

        planner = build_planner(config)
        current_config = base_config
        for iteration in range(1, max_iterations + 1):
            iter_dir = run_dir / f"iteration_{iteration:03d}"
            iter_dir.mkdir()
            progress.advance(f"iteration {iteration}/{max_iterations}: planning")
            context = make_context(config, program, task, goal, iteration, current_config, best, last, history)
            candidate = planner.propose(context)
            append_jsonl(run_dir / "planner_calls.jsonl", {"iteration": iteration, "candidate": candidate})
            write_json(iter_dir / "planner_output.json", candidate)

            progress.advance(f"iteration {iteration}/{max_iterations}: validating")
            validation = validate_candidate(candidate, config["search_space"], config.get("constraints", {}), current_config)
            write_json(iter_dir / "validation.json", validation)
            if not validation["valid"]:
                decision = {
                    "iteration": iteration,
                    "candidate": candidate,
                    "validation": validation,
                    "decision": "rollback",
                    "reason": "candidate failed validation",
                }
                append_jsonl(run_dir / "decisions.jsonl", decision)
                append_jsonl(run_dir / "errors.jsonl", decision)
                progress.advance(f"iteration {iteration}/{max_iterations}: invalid candidate rollback")
                progress.advance(f"iteration {iteration}/{max_iterations}: skipped")
                continue

            candidate_config = apply_changes(current_config, candidate["changes"])
            fingerprint = config_fingerprint(candidate_config)
            if fingerprint in seen_configs:
                decision = {
                    "iteration": iteration,
                    "candidate": candidate,
                    "validation": validation,
                    "decision": "reject",
                    "reason": "candidate repeats a previously tested configuration",
                }
                append_jsonl(run_dir / "decisions.jsonl", decision)
                append_jsonl(run_dir / "errors.jsonl", decision)
                progress.advance(f"iteration {iteration}/{max_iterations}: duplicate candidate rejected")
                progress.advance(f"iteration {iteration}/{max_iterations}: skipped")
                continue
            seen_configs.add(fingerprint)
            candidate_path = iter_dir / "candidate.yaml"
            write_yaml(candidate_path, candidate_config)
            progress.advance(f"iteration {iteration}/{max_iterations}: running train/eval")
            candidate_task_run_id = f"{task_name}/{run_id}/iteration_{iteration:03d}/baseline"
            returncode, summary_path = run_task_experiment(task, candidate_path, candidate_task_run_id, run_dir / "tool_calls.jsonl", progress)
            summary = load_summary(summary_path)
            reflection = reflect_on_result(summary, best, config["objective"])
            write_json(iter_dir / "reflection.json", reflection)
            decision, reason = choose_decision(summary, best, config["objective"])
            item = {
                "iteration": iteration,
                "candidate": candidate,
                "candidate_config": str(candidate_path),
                "summary_path": str(summary_path),
                "returncode": returncode,
                "summary": summary,
                "reflection": reflection,
                "decision": decision,
                "reason": reason,
            }
            append_jsonl(run_dir / "decisions.jsonl", item)
            history.append(item)
            last = {"label": f"iteration_{iteration:03d}", "config": candidate_config, "summary": summary}
            if decision == "accept":
                best = last
                current_config = candidate_config
            progress.advance(f"iteration {iteration}/{max_iterations}: complete {decision} mean_score={summary.get('mean_score')}")
            state.update({
                "iteration": iteration,
                "best": {"label": best["label"], "summary": best["summary"]} if best else None,
                "history": [{"iteration": h["iteration"], "decision": h["decision"], "reason": h["reason"]} for h in history],
            })
            write_state(run_dir, state)

        state["status"] = "success"
        state["duration_seconds"] = time.time() - started
        state["best"] = {"label": best["label"], "summary": best["summary"]} if best else None
        state["version_control"] = maybe_commit_best_config(config, task_name, task_dir, best)
        write_state(run_dir, state)
        generate_report(run_dir, goal, config, baseline, history, best)
        progress.finish("complete")
    except Exception as exc:
        progress.line("Agent failed; writing error state.")
        state["status"] = "error"
        state["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        append_jsonl(run_dir / "errors.jsonl", state["error"])
        write_state(run_dir, state)
        raise

    print(json.dumps({
        "run_dir": str(run_dir),
        "state_path": str(run_dir / "state.json"),
        "report_path": str(run_dir / "report.md"),
        "status": state["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
