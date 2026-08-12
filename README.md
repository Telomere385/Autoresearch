# Mini AutoResearch RL Agent

This repository is a small AutoResearch-style framework for interface-clear reinforcement-learning tasks. The included task is `flappy_qlearning`, a Flappy Bird tabular Q-learning experiment.

## Layout

```text
agent.py                 Main AutoResearch loop.
planner/                 LLM planner and provider adapters.
scripts/                 Maintenance CLIs for onboarding, reports, and Git helpers.
tasks/flappy_qlearning/  Runnable Flappy Q-learning task.
tasks/template_rl_task/  Template for adding another task.
docs/                    Operating protocol and project notes.
runs/                    Generated experiment outputs.
reports/                 Generated dashboards and packages.
```

Legacy files and old experiment outputs should not live in the active tree. Keep reproducible task code under `tasks/`, and keep generated artifacts under `runs/` or `reports/`.

## Install

```powershell
pip install -r requirements.txt
pip install -r tasks/flappy_qlearning/requirements.txt
```

## Configure The Planner

For DashScope/Kimi, set the API key in the shell:

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
```

The active task config is:

```text
tasks/flappy_qlearning/agent.yaml
```

Its planner block should look like:

```yaml
planner:
  type: llm
  provider: openai
  model: kimi/kimi-k3
  api_key_env: DASHSCOPE_API_KEY
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  timeout_seconds: 60
  max_retries: 3
  retry_initial_seconds: 2.0
  temperature: 0.2
```

`provider: openai` means OpenAI-compatible `/chat/completions`; it is used for DashScope compatible mode as well.

## Run

Use a fresh `--run-id` each time:

```powershell
python agent.py --task flappy_qlearning --run-id kimi_test_001
```

Run without network using the mock planner:

```powershell
python agent.py --task flappy_qlearning --config tasks/flappy_qlearning/agent_smoke.yaml --run-id smoke_001
```

Generate dashboard and report package:

```powershell
python scripts/visualize.py --task flappy_qlearning --run-id kimi_test_001
python scripts/package_report.py --task flappy_qlearning --run-id kimi_test_001
```

## Add A Task

Place a project under:

```text
tasks/<task_name>/project/
```

Generate the task adapter:

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
python scripts/onboard_task.py --task <task_name> --provider openai --model kimi/kimi-k3 --api-key-env DASHSCOPE_API_KEY --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --force
```

Then run:

```powershell
python agent.py --task <task_name>
```

## Task Contract

Each task must provide one of these layouts:

```text
tasks/<task_name>/task.yaml
tasks/<task_name>/agent.yaml
tasks/<task_name>/config.yaml
tasks/<task_name>/goal.md
tasks/<task_name>/run.py
```

or:

```text
tasks/<task_name>/manifest/task.yaml
tasks/<task_name>/configs/agent.yaml
tasks/<task_name>/configs/experiment.yaml
tasks/<task_name>/configs/goal.md
tasks/<task_name>/runner/run.py
```

The runner must accept:

```powershell
python <task_runner> --config <candidate_config> --run-id <task>/<run_id>/<phase>
```

and write:

```text
runs/<task>/<run_id>/<phase>/summary.json
```

Minimum summary fields:

```json
{
  "status": "success",
  "mean_score": 0.0,
  "max_score": 0.0,
  "mean_reward": 0.0,
  "total_training_time": 0.0
}
```
