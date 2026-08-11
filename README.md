# Mini AutoResearch Flappy Bird

This is a compact, offline-first Mini AutoResearch project around the existing Flappy Bird Q-learning experiment. The active project surface is intentionally small: the Agent reads a goal and YAML config, proposes bounded parameter changes, runs real train/eval jobs, checks JSON metrics, and records accept/reject/rollback decisions.

## Structure

```text
Lab2_new/
  agent.py             AutoResearch controller and decision loop.
  baseline.py          Single-seed train/eval runner.
  experiment.py        Single train or eval run from YAML.
  package_report.py    Packages one run into a report folder.
  visualize.py         Renders a local HTML performance dashboard.
  configs/
    agent.yaml         Main AutoResearch configuration.
    experiment.yaml    Default single-experiment configuration.
  goals/
    default_goal.md    Natural language research goal.
  planner/
    rules.py           Default offline planner.
    llm.py             Optional structured LLM planner adapter.
    providers.py       Optional provider interfaces.
  src/
    flappyq.py         Flappy Bird environment and Q-learning loop.
  assets/              Pygame sprites and audio required by the environment.
  data/q_tables/
    Qvals.npy          Initial Q-table.
  runs/                Generated experiment outputs.
  reports/             Generated report packages.
  archive/             Legacy course files, old configs, old outputs, and media.
```

## Install

```powershell
pip install -r requirements.txt
```

## One-Command Agent Run

```powershell
python agent.py --config configs/agent.yaml
```

The default planner is `rules` with `provider: disabled`, so no API key is required. Optional LLM providers can be configured in `configs/agent.yaml`; their output is constrained to structured candidate parameter changes and validated against the YAML search space before any experiment runs.

## Single Experiment

Train:

```powershell
python experiment.py --mode train --config configs/experiment.yaml --run-id manual_train
```

Evaluate a trained Q-table without exploration:

```powershell
python experiment.py --mode eval --config configs/experiment.yaml --run-id manual_eval --set input_q_table=runs/manual_train/q_table.npy
```

## Outputs

Each Agent run writes:

```text
runs/<run_id>/
  config.yaml
  goal.md
  state.json
  decisions.jsonl
  planner_calls.jsonl
  tool_calls.jsonl
  errors.jsonl
  report.md
  iteration_001/
  iteration_002/
  iteration_003/
```

Package a completed run:

```powershell
python package_report.py --run-id <run_id> --config-used configs/agent.yaml
```

Render a visual performance dashboard:

```powershell
python visualize.py --run-id <run_id>
```

The dashboard is written to:

```text
reports/<run_id>/dashboard.html
```

## Archive

`archive/` contains material moved out of the active project surface: original course controllers, old plotting scripts, demo media, old configs, historical Q-tables, reward arrays, and previous run/report outputs. It is kept for reference and is not required to run the Mini AutoResearch Agent.
