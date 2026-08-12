# Mini AutoResearch Agent Harness

This repository implements a small LLM Agent that improves a real Flappy Bird Q-learning experiment. The LLM receives a natural-language goal, creates a multi-step plan, selects native function tools, edits run-local configuration, launches Python subprocesses, inspects metrics and logs, decides whether to accept or roll back each candidate, and writes the final report.

Python is the harness, not the planner. It executes the requested tools, restricts filesystem and command access, independently validates metrics, verifies rollback snapshots, enforces budgets, and records the complete interaction.

## Install

Python 3.10 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
```

The Flappy Bird task sets SDL video and audio drivers to `dummy` for headless execution.

## Configure

The default configuration is [configs/autoresearch.yaml](configs/autoresearch.yaml). It contains:

- OpenAI-compatible provider, model, API endpoint, timeout, and retry settings;
- natural-language goal file, task command, data/Q-table path, and metric path;
- minimum/maximum candidate experiments, tool-call, failure, wall-time, and subprocess budgets;
- baseline configuration and allowed candidate parameter values;
- readable paths, the run-local writable path, and the command allowlist;
- the requirement for a real failure-and-recovery event.

Set the configured API key environment variable:

```powershell
$env:DASHSCOPE_API_KEY="your-api-key"
```

The default `kimi/kimi-k3` setup uses OpenAI-compatible native Function Calling. The client preserves Kimi's `reasoning_content` in subsequent messages and disables parallel tool calls.

## Run

Start the complete workflow with one main command:

```powershell
python main.py --config configs/autoresearch.yaml --run-id research_agent_001
```

Progress is printed live to `stderr` while the final machine-readable result remains on `stdout`:

```text
[13:41:10] PLAN    submitted 6 steps
[13:41:20] LLM     call 2 completed in 10.3s, tokens=1842
[13:41:20] TOOL    call 2 read_file tasks/flappy_qlearning/goal.md
[13:43:33] EXP     baseline started, timeout=300s
[13:43:48] EXP     baseline finished in 15.2s, returncode=0, mean_score=5.2
```

An LLM request or experiment that takes longer than 15 seconds emits a heartbeat every 15 seconds. Use `--quiet` to disable progress or `--verbose` to include API retry and bounded output-tail diagnostics:

```powershell
python main.py --config configs/autoresearch.yaml --run-id research_agent_001 --verbose
```

Progress never prints API keys, complete prompts, file contents, or full model reasoning.

Override the goal directly from the command line:

```powershell
python main.py --config configs/autoresearch.yaml --run-id research_agent_002 --goal "在五次实验内提高 mean_score，至少完成三轮候选实验，并记录失败恢复过程"
```

Every run ID must be new; the Agent never overwrites an existing run.

Run the regression and end-to-end harness tests with:

```powershell
python -m unittest discover -s tests -v
```

The end-to-end test uses a scripted test double only to make protocol assertions deterministic. It still launches three real temporary Python subprocess experiments and derives their metrics from written configurations. The production entry point has no scripted proposals or prewritten experiment results.

## Agent and Tool Loop

The model receives native tool schemas and must call exactly one tool per turn:

- `submit_plan` and `update_plan` create and maintain an explicit multi-step plan;
- `list_files` and `read_file` inspect the goal, source, configuration, data metadata, logs, and metrics;
- `write_file` writes only inside `runs/<run_id>/` and records before/after hashes;
- `run_command` uses `subprocess.run(..., shell=False)` and only permits the configured Python experiment;
- `evaluate_result` records the LLM's evidence-based accept/rollback decision and compares it with an independent metric check;
- `restore_snapshot` performs and verifies a requested rollback;
- `finish` validates minimum iterations, plan completion, recovery evidence, and the LLM-authored report.

The baseline is not launched by a hidden Python workflow. The LLM must inspect inputs, write `current_config.yaml`, and call `run_command` itself. Candidate results are not precomputed or hard-coded.

At least three candidate experiments are required; the baseline does not count. The default maximum is five so the Agent can obtain real failure/recovery evidence. A normal finish requires a measured candidate regression or failed experiment followed by a verified snapshot rollback. The prompt explicitly forbids intentionally corrupting a command or configuration to manufacture failure. Safety limits can still stop an unhealthy run early.

## Independent Verification

The harness does not assume that a command or LLM conclusion is correct. It checks:

- subprocess timeout and return code;
- metrics file existence and valid JSON;
- task `status` and a finite objective metric;
- candidate changes against the configured search space;
- duplicate configurations;
- candidate metrics against the canonical best result;
- rollback contents by reading back the saved snapshot;
- final report sections, all iteration references, and the exact best metric value.

An invalid tool request is returned to the model as a real tool error so it can reflect and correct its next action. An improving candidate may be conservatively rolled back by the LLM, but a declining or invalid candidate cannot be accepted against the independent checker.

## Execution Records

Each run is fully auditable:

```text
runs/<run_id>/
  run_config.yaml
  current_config.yaml
  plan.json
  state.json
  trajectory.jsonl
  messages.jsonl
  report.md
  llm_calls/call_XXX/
    request.json
    response.json
  tool_calls/call_XXX_<tool>/
    arguments.json
    result.json
  snapshots/
  baseline/
    execution.json
    metrics.json
    stdout.log
    stderr.log
    raw/
  iteration_01/
  iteration_02/
  iteration_03/
```

`messages.jsonl` contains the complete LLM/tool conversation. `trajectory.jsonl` contains decisions, arguments, outputs, errors, recovery actions, and next actions. API keys are read only from the environment and are never persisted.

## Termination

Normal completion uses `agent_finish`. Safety exits include:

- `tool_call_budget_exhausted`;
- `too_many_failures`;
- `wall_time_budget_exhausted`;
- `unrecoverable_error`.

A safety exit before all requirements are met is reported as `failed` or `incomplete_requirements`, never as a successful research result.
