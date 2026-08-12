# Mini AutoResearch

This repository contains a small research agent for iterating real Flappy Bird Q-learning experiments. The LLM chooses only the next bounded hyperparameter change. Python validates the proposal, edits the active YAML, runs the task, validates metrics, compares results, and accepts or rolls back the candidate.

## Structure

```text
main.py                         Single command-line entry point
autoresearch/agent.py           Complete observe-decide-act-evaluate-update loop
autoresearch/planner.py         One-experiment LLM request and JSON parsing
autoresearch/execution.py       Config edits, subprocess execution, validation, comparison
autoresearch/report.py          Report generated only from persisted state
configs/autoresearch.yaml       Runtime, model, budget, task, objective, and search configuration
configs/smoke.yaml              Fast offline verification configuration
tasks/flappy_qlearning/run.py   Train/evaluate task adapter
tasks/flappy_qlearning/src/     Existing Flappy Bird Q-learning implementation
tools/mock_llm_planner.py       State-dependent offline planner for smoke tests
runs/                           Generated run artifacts
```

The control loop is intentionally in one file. `state.json` is the canonical state; `trajectory.jsonl` is the append-only event record.

## Install

```powershell
pip install -r requirements.txt
```

The task runs headlessly. Its code sets the SDL video and audio drivers to `dummy` unless they are already configured.

## Configure

Edit [configs/autoresearch.yaml](configs/autoresearch.yaml). It is the only runtime configuration and contains:

- planner provider, model, base URL, API key environment variable, timeout, and retries
- goal and experiment command paths
- experiment, Agent-step, failure, wall-time, and subprocess limits
- metric, optimization direction, optional target, and tie-breaker
- baseline experiment and allowed parameter values

Secrets are read from the configured environment variable and do not belong in YAML. The default DashScope-compatible setup uses:

```powershell
$env:DASHSCOPE_API_KEY="your-api-key"
```

## Run

Run the real LLM-planned workflow:

```powershell
python main.py --config configs/autoresearch.yaml --run-id research_001
```

Run the fast offline workflow. The local planner reads the accumulated real history before every proposal; it is not a precomputed sweep.

```powershell
python main.py --config configs/smoke.yaml --run-id smoke_001
```

Run regression tests:

```powershell
python -m unittest discover -s tests -v
```

Use a fresh run ID because the Agent refuses to overwrite an existing run.

## Control Flow

For every run, `autoresearch.agent.run_agent`:

1. Loads the goal and the single runtime config.
2. Writes and executes the baseline configuration.
3. Builds planner context from canonical state, including all prior outcomes and failures.
4. Requests and validates one proposal.
5. Snapshots `current_config.yaml`, applies the candidate, and reads it back.
6. Executes the configured command with a timeout and captures its exit code and output.
7. Requires a successful JSON result with a finite configured metric.
8. Compares candidate and best metrics deterministically.
9. Keeps an accepted config or restores and verifies the pre-experiment config.
10. Persists state and trajectory, checks explicit limits, and writes `report.md`.

Planning and engineering failures consume bounded Agent steps but not real experiment iterations. A worse valid experiment is recorded as rejected and rolled back.

## Artifacts

```text
runs/<run_id>/
  run_config.yaml
  current_config.yaml
  state.json
  trajectory.jsonl
  report.md
  baseline/
    config.yaml
    execution.json
    metrics.json
    stdout.log
    raw/
  planning_attempts/step_01/
    input.json
    raw_output.txt
    proposal.json
    validation.json
  iteration_01/
    proposal.json
    before_config.yaml
    after_config.yaml
    execution.json
    metrics.json
    stdout.log
    raw/
```

The task's detailed train/eval results, Q-table, and logs remain under each `raw/` directory. Reports never fill in missing metrics.

## Stop Reasons

The persisted stop reason is one of:

- `iteration_budget_exhausted`
- `agent_step_budget_exhausted`
- `too_many_failures`
- `wall_time_budget_exhausted`
- `target_reached`
- `unrecoverable_error`

Every planner and experiment timeout is bounded. The task adapter returns a non-zero exit code when train/evaluation fails.
