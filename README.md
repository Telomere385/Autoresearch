# Mini AutoResearch RL Agent

This repository is a compact AutoResearch-style framework for bounded reinforcement-learning experiments. It is designed to be operated by Codex: the project defines the task interface, editable surface, budget, commands, and artifact format; Codex reads the user's natural-language goal, runs real experiments, evaluates metrics, and writes the final report from recorded artifacts.

The included task adapter is Flappy Bird tabular Q-learning.

## Core Files

```text
PROGRAM.md              Codex operating protocol.
configs/task.yaml       Generic RL task interface for the current adapter.
configs/agent.yaml      LLM planner, budget, objective, and search space.
configs/experiment.yaml Default task experiment config.
goals/default_goal.md   Example natural-language goal.

agent.py                AutoResearch controller.
baseline.py             Single-seed train/eval task runner.
experiment.py           Single train or eval run.
visualize.py            HTML dashboard renderer.
package_report.py       Artifact packager.

planner/
  llm.py                Structured LLM planner.
  providers.py          OpenAI, local endpoint, local command providers.

src/flappyq.py          Current RL environment and Q-learning implementation.
data/q_tables/Qvals.npy Initial Q-table for the current task.
```

## Install

```powershell
pip install -r requirements.txt
```

## Run With OpenAI Planner

```powershell
$env:OPENAI_API_KEY="your_api_key"
python agent.py --config configs/agent.yaml
```

The LLM may only propose parameter changes from `configs/agent.yaml::search_space`. The Agent validates every candidate before running it.

## No-Network Interface Test

```powershell
python agent.py --config configs/agent_smoke.yaml
```

This uses `provider: local_command` and `tools/mock_llm_planner.py`. It tests the same structured planner interface without a remote API call.

## Codex Workflow

When using Codex directly, start from:

```text
PROGRAM.md
configs/task.yaml
configs/agent.yaml
```

Then run:

```powershell
python agent.py --config configs/agent.yaml --run-id <run_id>
python visualize.py --run-id <run_id>
python package_report.py --run-id <run_id> --config-used configs/agent.yaml
```

For the final written report, use:

```text
prompts/final_report.md
```

Codex should read the run artifacts and write `reports/<run_id>/codex_report.md`.

## Outputs

```text
runs/<run_id>/
  program.md
  task.yaml
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

reports/<run_id>/
  dashboard.html
  README.md
  technical_report.md
  run_record.md
  artifacts/
```

## Adapting To Another RL Task

To adapt this framework to another interface-clear RL task:

1. Replace or add a task runner equivalent to `baseline.py`.
2. Ensure the runner accepts `--config` and `--run-id`.
3. Ensure it writes `runs/<run_id>/summary.json`.
4. Update `configs/task.yaml` with the command, output paths, metric names, and editable surface.
5. Update `configs/agent.yaml::base_experiment` and `search_space`.
6. Keep results in the same JSON artifact format.

The controller does not need to know the RL algorithm internals as long as the task adapter follows the interface.

## Current Limitations

- The included adapter is still Flappy Bird Q-learning, not a broad RL benchmark suite.
- The framework optimizes bounded YAML parameters, not arbitrary source code.
- Final narrative reports are intended to be written by Codex from artifacts, not treated as scientific claims.
- Single-seed evaluation is fast and simple but not statistically rigorous.
