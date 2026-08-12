import json
from pathlib import Path


def generate_report(state, path):
    objective = state["objective"]
    metric = objective["metric"]
    baseline = state.get("baseline") or {}
    lines = [
        "# Mini AutoResearch Report",
        "",
        "## Research Goal",
        state["goal"].strip(),
        "",
        "## Baseline",
        f"- Status: {baseline.get('status', 'not run')}",
        f"- {metric}: {_metric(baseline.get('metrics'), metric)}",
        f"- Configuration: `{json.dumps(baseline.get('config', {}), sort_keys=True)}`",
        "",
        "## Experiment Trajectory",
    ]
    if not state["history"]:
        lines.append("No candidate experiment completed.")
    for item in state["history"]:
        lines.extend([
            "",
            f"### Iteration {item['iteration']}",
            f"- Hypothesis: {item['proposal']['hypothesis']}",
            f"- Changes: `{json.dumps(item['proposal']['changes'], sort_keys=True)}`",
            f"- Execution: {item['execution_status']}",
            f"- {metric}: {_metric(item.get('metrics'), metric)}",
            f"- Decision: {item['decision']}",
            f"- Reason: {item['reason']}",
            f"- Recovery: {item.get('recovery_action') or 'none'}",
        ])

    lines.extend(["", "## Failures And Recovery"])
    if state["failures"]:
        for failure in state["failures"]:
            lines.append(
                f"- Step {failure.get('agent_step')}, {failure['type']}: "
                f"{failure['message']} Recovery: {failure.get('recovery_action', 'none')}."
            )
    else:
        lines.append("- No planning or execution failure was recorded.")
    rejected = [item for item in state["history"] if item["decision"] != "accept"]
    for item in rejected:
        lines.append(f"- Iteration {item['iteration']} was {item['decision']}; {item['recovery_action']}.")

    lines.extend([
        "",
        "## Best Result",
        f"- Source: {state.get('best_source') or 'none'}",
        f"- {metric}: {_metric(state.get('best_metrics'), metric)}",
        f"- Configuration: `{json.dumps(state.get('best_config') or {}, sort_keys=True)}`",
        "",
        "## Termination",
        f"- Stop reason: {state.get('stop_reason') or 'unknown'}",
        f"- Real candidate experiments: {state['iteration']} / {state['limits']['max_iterations']}",
        f"- Agent steps: {state['agent_steps']} / {state['limits']['max_agent_steps']}",
        "",
        "## Limitations",
        "- Results use one configured seed and are not a statistical performance claim.",
        "- The planner can only modify the explicitly configured hyperparameter search space.",
        "- Short Flappy Bird runs may produce zero scores; mean reward is retained as a diagnostic.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric(metrics, name):
    return "n/a" if not metrics or name not in metrics else metrics[name]
