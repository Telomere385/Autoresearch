# RL Task Template

Copy this folder to `tasks/<your_task_name>/` and replace the placeholders.

## Structure

```text
tasks/template_rl_task/
  README.md
  manifest/
    task.yaml          # Task contract: runner command, metrics, editable parameters.
  configs/
    agent.yaml         # Agent budget, planner provider, objective, search space.
    experiment.yaml    # Baseline experiment config passed to the runner.
    goal.md            # Natural-language research goal.
  runner/
    run.py             # Required task entrypoint.
```

## Required Runner Contract

`runner/run.py` must accept:

```powershell
python tasks/<your_task_name>/runner/run.py --config <candidate.yaml> --run-id <task>/<run_id>
```

It must run a real train/eval experiment and write:

```text
runs/<task>/<run_id>/summary.json
```
