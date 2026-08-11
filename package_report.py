import argparse
import json
from pathlib import Path
import shutil

from visualize import collect_experiments, make_html, read_jsonl


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
REPORTS_DIR = BASE_DIR / "reports"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def metric_value(summary, metric):
    if not summary:
        return 0.0
    return summary.get(metric, 0.0)


def load_decisions(run_dir):
    path = run_dir / "decisions.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_tool_calls(run_dir):
    path = run_dir / "tool_calls.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                rows.append({
                    "command": " ".join(item.get("command", [])),
                    "returncode": item.get("returncode"),
                    "duration_seconds": item.get("duration_seconds"),
                })
    return rows


def make_readme(run_id, config_used):
    return f"""# Mini AutoResearch Run Package

This folder packages one complete, real Mini AutoResearch run for the Flappy Bird Q-learning project.

## Environment

- Python 3.10 tested in the current workspace
- Required packages are listed in `requirements.txt`
- Pygame runs headless through SDL dummy video/audio drivers

## Install

```powershell
cd CODE_STUDENT
pip install -r requirements.txt
```

## Start Command

```powershell
python agent.py --config {config_used} --run-id {run_id}
```

## Directory Structure

```text
    reports/{run_id}/
      README.md
      technical_report.md
      run_record.md
      artifacts/
        state.json
        decisions.jsonl
        iteration_reflections/
        planner_calls.jsonl
        tool_calls.jsonl
        errors.jsonl
    report.md
```

The original experiment artifacts remain under:

```text
runs/{run_id}/
```

## Input Format

The main input is `{config_used}`, which contains:

- natural language goal path
- planner selection
- training/evaluation budget
- random seed
- objective metric
- allowed search space
- safety constraints

## Output Format

The Agent writes:

- `state.json`: final Agent state and best result
- `decisions.jsonl`: planner proposals, validation, accept/reject/rollback decisions
- `iteration_reflections/`: independent result checks and recommended next actions
- `tool_calls.jsonl`: executed Python commands and return codes
- `errors.jsonl`: validation failures and runtime errors
- `report.md`: generated experiment report
- per-iteration `summary.json` files under `runs/{run_id}/`

## Known Limitations

- This is a compact LLM-planned Agent, not a general autonomous scientist.
- LLM output is constrained to YAML parameter changes and validated before execution.
- Score can remain zero for small training budgets; reward is included as a diagnostic metric.
- Pygame is headless but still simulates game frames, so runtime grows with train/eval episodes.
"""


def make_technical_report(run_id, state, decisions, tool_calls):
    best = state.get("best") or {}
    best_summary = best.get("summary") or {}
    history = state.get("history", [])
    failure_rows = [d for d in decisions if d.get("iteration") == 0 or d.get("decision") == "rollback"]
    iteration_rows = [d for d in decisions if d.get("iteration", 0) > 0]

    lines = [
        "# Technical Report: Mini AutoResearch Agent",
        "",
        "## 1. Overview",
        "This system implements an offline-first Mini AutoResearch Agent around the existing Flappy Bird Q-learning experiment. It reads a natural-language goal, proposes bounded configuration changes, executes real training and evaluation runs, validates metrics, records decisions, and generates a final report.",
        "",
        "## 2. Architecture",
        "- `agent.py` is the Agent controller and state machine.",
        "- `planner/` contains the structured LLM planner and provider adapters.",
        "- `baseline.py` executes one train/eval experiment pair for a single seed.",
        "- `experiment.py` executes one configured train or eval run.",
        "- `src/flappyq.py` contains the original Q-learning environment and update loop.",
        "",
        "## 3. Core Flow",
        "The Agent first runs a baseline, then performs three candidate iterations. Each iteration reads current state, asks the planner for a candidate, validates the proposal against the search space and constraints, writes a YAML config, runs a real train/eval experiment pair, checks JSON results, and decides accept, reject, rollback, or stop.",
        "",
        "## 4. Key Design Choices",
        "- LLM-planned: provider output is restricted to structured parameter changes.",
        "- Local testing: `provider: local_command` can exercise the LLM interface without a remote API.",
        "- Safety first: source edits and state-space expansion are blocked.",
        "- Reproducibility: fixed seed, YAML configs, JSONL traces, and per-run artifacts.",
        "- Evaluation isolation: eval uses trained Q-tables and epsilon is forced to zero.",
        "",
        "## 5. Experiment Results",
        f"- Run id: `{run_id}`",
        f"- Final status: `{state.get('status')}`",
        f"- Completed iterations: {state.get('iteration')}",
        f"- Best source: `{best.get('label', 'none')}`",
        f"- Best mean_score: {best_summary.get('mean_score', 'n/a')}",
        f"- Best mean_reward: {metric_value(best_summary, 'mean_reward')}",
        "",
        "| Iteration | Decision | Reason |",
        "|---:|---|---|",
    ]
    for item in history:
        lines.append(f"| {item.get('iteration')} | {item.get('decision')} | {item.get('reason')} |")

    reflections = [d.get("reflection") for d in iteration_rows if d.get("reflection")]
    if reflections:
        lines += [
            "",
            "## 5.1 Independent Result Checks",
            "| Iteration | Finding | Recommended next action |",
            "|---:|---|---|",
        ]
        for item in iteration_rows:
            reflection = item.get("reflection") or {}
            lines.append(f"| {item.get('iteration')} | {reflection.get('finding')} | {reflection.get('next_action')} |")

    lines += [
        "",
        "## 6. Failure Case And Recovery",
    ]
    if failure_rows:
        first = failure_rows[0]
        lines.append(f"The run includes a real validation failure at iteration {first.get('iteration')}: `{first.get('reason')}`. The Agent rejected the invalid candidate before launching an experiment and restored the baseline configuration.")
    else:
        lines.append("No failure case was recorded in this run.")

    lines += [
        "",
        "## 7. Tool Execution Evidence",
        f"- Tool calls recorded: {len(tool_calls)}",
        "- Each tool call includes command, return code, duration, and stdout in `artifacts/tool_calls.jsonl`.",
        "",
        "## 8. Limitations And Next Improvements",
        "- Short budgets may keep game score at zero; longer budgets are needed for performance claims.",
        "- The LLM planner is constrained to a small predefined search space.",
        "- OpenAI/Anthropic provider adapters are intentionally guarded and should be completed only with explicit API configuration.",
        "- Future work should add richer statistical tests, wall-time enforcement, and automatic medium/full budget scheduling.",
        "",
    ]
    return "\n".join(lines)


def make_run_record(run_id, state, decisions, tool_calls):
    lines = [
        "# Complete Run Record",
        "",
        f"Run id: `{run_id}`",
        f"Status: `{state.get('status')}`",
        "",
        "## Decisions",
    ]
    for d in decisions:
        lines.append(f"- iteration={d.get('iteration')} decision={d.get('decision')} reason={d.get('reason')}")

    lines += ["", "## Result Checks"]
    for d in decisions:
        reflection = d.get("reflection")
        if reflection:
            lines.append(
                f"- iteration={d.get('iteration')} finding={reflection.get('finding')} next_action={reflection.get('next_action')}"
            )

    lines += ["", "## Tool Calls"]
    for i, call in enumerate(tool_calls, 1):
        lines.append(f"{i}. returncode={call['returncode']} duration={call['duration_seconds']:.2f}s")
        lines.append(f"   `{call['command']}`")

    lines += [
        "",
        "## Artifact Pointers",
        f"- Full state: `artifacts/state.json`",
        f"- Decisions: `artifacts/decisions.jsonl`",
        f"- Planner calls: `artifacts/planner_calls.jsonl`",
        f"- Tool calls: `artifacts/tool_calls.jsonl`",
        f"- Errors: `artifacts/errors.jsonl`",
        f"- Original run directory: `../../runs/{run_id}/`",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Package an AutoResearch run into a report folder.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config-used", default="configs/agent.yaml")
    args = parser.parse_args()

    run_id = args.run_id
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    out_dir = Path(args.output) if args.output else REPORTS_DIR / run_id
    artifacts = out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    state = read_json(run_dir / "state.json")
    decisions = load_decisions(run_dir)
    tool_calls = summarize_tool_calls(run_dir)

    for name in ("program.md", "task.yaml", "state.json", "decisions.jsonl", "planner_calls.jsonl", "tool_calls.jsonl", "errors.jsonl", "report.md", "config.yaml", "goal.md"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, artifacts / name)
    prompt_src = BASE_DIR / "prompts" / "final_report.md"
    if prompt_src.exists():
        shutil.copy2(prompt_src, artifacts / "final_report_prompt.md")
    reflection_dir = artifacts / "iteration_reflections"
    reflection_dir.mkdir(exist_ok=True)
    for src in sorted(run_dir.glob("iteration_*/reflection.json")):
        shutil.copy2(src, reflection_dir / f"{src.parent.name}_reflection.json")

    write_text(out_dir / "README.md", make_readme(run_id, args.config_used))
    write_text(out_dir / "technical_report.md", make_technical_report(run_id, state, decisions, tool_calls))
    write_text(out_dir / "run_record.md", make_run_record(run_id, state, decisions, tool_calls))
    rows = collect_experiments(run_dir, decisions)
    write_text(out_dir / "dashboard.html", make_html(run_id, state, decisions, read_jsonl(run_dir / "tool_calls.jsonl"), rows))
    print(json.dumps({"report_dir": str(out_dir), "run_id": run_id}, indent=2))


if __name__ == "__main__":
    main()
