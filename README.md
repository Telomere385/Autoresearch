# Mini AutoResearch Agent Harness

Mini AutoResearch is a compact LLM-agent harness for running real, bounded machine-learning experiments. In the included task, the agent improves a Flappy Bird Q-learning policy by inspecting the project, planning a sequence of experiments, editing run-local configuration, launching training and evaluation processes, comparing measured results, and documenting its findings.

The LLM decides what to investigate and which experiment to try next. The Python harness controls what may actually happen: it validates every tool request, restricts file and command access, verifies metrics independently, enforces experiment budgets, performs checked rollbacks, and records the complete run for later audit.

## Key Properties

- **Real experiments:** candidates are trained and evaluated by Python subprocesses; results are neither simulated nor precomputed.
- **Bounded autonomy:** the search space, filesystem access, command execution, experiment count, tool-call count, and runtime are explicitly limited.
- **Independent verification:** process status, metric files, candidate configurations, comparisons, rollbacks, and the final report are checked by deterministic code.
- **Recoverable iteration:** each candidate starts from a snapshot of the last accepted configuration, allowing regressions to be rolled back and verified.
- **State-aware context:** every model request receives an authoritative state summary plus a bounded window of recent complete tool-call turns.
- **Complete traceability:** model messages, tool calls, configurations, logs, metrics, state transitions, and reports are persisted under a unique run directory.

For a detailed architecture and experiment analysis, see [Report.md](Report.md).

## Quick Start

Python 3.10 or newer is recommended.

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

The Flappy Bird task automatically uses SDL's `dummy` video and audio drivers, so experiments can run without a display or audio device.

### 2. Configure the provider

The default runtime configuration is [configs/autoresearch.yaml](configs/autoresearch.yaml). It defines:

- the OpenAI-compatible provider, model, endpoint, timeout, and retry policy;
- the model-facing context window and state-summary controls;
- the natural-language goal, experiment command, input data, and metrics path;
- the optimization objective, baseline, and allowed hyperparameter values;
- candidate, tool-call, failure, wall-clock, and subprocess budgets;
- readable paths, the run-local writable path, and the command allowlist;
- the requirement to demonstrate a measured failure or regression followed by a verified rollback.

Set the API key through the environment variable named in the configuration:

```powershell
$env:DASHSCOPE_API_KEY="your-api-key"
```

The default configuration uses `kimi/kimi-k3` through DashScope's OpenAI-compatible Chat Completions endpoint and the official OpenAI Python SDK. SDK timeouts and retries are configurable. Provider-specific `reasoning_content` returned by Kimi is preserved in later messages, and parallel tool calls are disabled.

### 3. Start a run

```powershell
python main.py --config configs/autoresearch.yaml --run-id research_agent_001
```

Each run ID must be new. The harness creates `runs/<run_id>/` with `exist_ok=False` and never overwrites an existing run.

Live progress is written to `stderr`, while the final machine-readable result is written to `stdout`:

```text
[13:41:10] PLAN    submitted 6 steps
[13:41:20] LLM     call 2 completed in 10.3s, tokens=1842
[13:41:20] TOOL    call 2 read_file tasks/flappy_qlearning/goal.md
[13:43:33] EXP     baseline started, timeout=300s
[13:43:48] EXP     baseline finished in 15.2s, returncode=0, mean_score=5.2
```

Long-running model requests and experiments emit a heartbeat every 15 seconds. Use `--quiet` to suppress progress or `--verbose` to include bounded output-tail and SDK error diagnostics:

```powershell
python main.py --config configs/autoresearch.yaml --run-id research_agent_002 --verbose
```

Progress output does not include API keys, complete prompts, file contents, or full model reasoning.

To override the configured goal for one run:

```powershell
python main.py --config configs/autoresearch.yaml --run-id research_agent_003 --goal "在五次实验内提高 mean_score，至少完成三轮候选实验，并记录失败恢复过程"
```

## How the Agent Works

The agent follows an explicit plan–act–observe–validate loop:

1. It reads the research goal and submits a multi-step plan.
2. It inspects the runtime configuration, task code, data metadata, metrics, and logs as needed.
3. It writes the exact baseline configuration to the current run directory and launches the baseline explicitly.
4. It proposes candidates using only values declared in the configured search space.
5. It runs each candidate, inspects the measured result, and requests either acceptance or rollback.
6. The harness independently checks the decision and verifies any requested snapshot restoration.
7. The agent updates its plan, writes the final report, and requests completion.

Before each model call, the harness builds a compact request context from the fixed system prompt and original goal, up to four recent complete assistant/tool turn groups, and a machine-generated JSON `AUTHORITATIVE_STATE_SUMMARY`. The recent groups are limited to 24,000 serialized characters by default and are removed oldest-first without splitting an assistant tool call from its corresponding tool result. The summary exposes current plan progress, accepted and pending configurations, best metrics, the last experiment result, a compact latest-tool observation, completion requirements, failures, and budgets directly from authoritative state.

These controls are configured under `context`:

```yaml
context:
  enabled: true
  recent_turns: 4
  max_recent_chars: 24000
```

Set `enabled: false` to restore full-history model requests for compatibility or diagnosis. This compaction is character-based and does not replace model-specific token budgeting.

The model must issue exactly one native tool call per turn. The available tools are:

| Tool | Purpose |
|---|---|
| `submit_plan` | Create the required multi-step research plan before other work begins. |
| `update_plan` | Record step status, supporting evidence, and the next action. |
| `list_files` | List bounded file metadata under an approved read root. |
| `read_file` | Read a bounded UTF-8 slice from an approved path. |
| `write_file` | Write a size-limited UTF-8 artifact inside the current run directory. |
| `run_command` | Launch the configured Python experiment with validated arguments and `shell=False`. |
| `evaluate_result` | Submit an evidence-based accept/rollback decision for the pending candidate. |
| `restore_snapshot` | Restore and read-back verify the pre-candidate configuration snapshot. |
| `finish` | Complete the run after all plan, experiment, recovery, and report checks pass. |

The baseline is not hidden inside a preprogrammed optimization workflow: the agent must inspect the inputs, write `current_config.yaml`, and call `run_command` itself. Candidate proposals and outcomes are likewise not hard-coded.

## Safety and Independent Verification

The harness treats model output and subprocess output as evidence to verify, not as trusted truth.

Before and after an experiment, it checks:

- read and write paths against run-scoped allowlists;
- the executable, script, configuration path, run ID, and timeout;
- candidate changes against the configured search space;
- full-configuration fingerprints to reject duplicate candidates;
- subprocess timeout and return code;
- metrics-file existence, valid JSON, successful task status, and a finite objective metric;
- candidate metrics against the canonical best result;
- rollback contents by reading back the restored snapshot;
- final-report sections, iteration references, recovery evidence, and the exact best metric value.

A candidate that improves the objective may be conservatively rolled back by the LLM, but a declining or invalid candidate cannot be accepted against the independent checker. Invalid tool requests are returned as structured errors so the agent can correct them within its remaining failure and tool-call budgets.

The default protocol requires at least three and at most five candidate experiments; the baseline does not count. Normal completion also requires a genuine measured regression or failed experiment followed by a verified rollback. The system prompt explicitly forbids corrupting a command or configuration to manufacture this evidence.

## Run Artifacts

Every run is stored under a unique directory:

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
    context.json
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

The main records serve different purposes:

- `state.json` is the latest structured state-machine snapshot.
- `messages.jsonl` contains the complete system, user, assistant, and tool conversation.
- `trajectory.jsonl` records decisions, arguments, results, errors, recovery actions, and terminal events.
- `llm_calls/` preserves the exact compact request and raw response for each model call; `context.json` records the authoritative summary and compaction statistics used to construct that request.
- `tool_calls/` preserves the arguments and structured result for each tool call.
- baseline and iteration directories contain configurations, execution metadata, metrics, logs, and raw training/evaluation artifacts.

API keys are read only from the environment and are not persisted by the harness. Because run artifacts contain complete prompts and tool observations, review them before publishing or sharing them externally.

## Testing

Run the unit and end-to-end harness tests with:

```powershell
python -m unittest discover -s tests -v
```

The end-to-end test replaces only the network model with a scripted test double so protocol assertions remain deterministic. It still uses the production control loop, launches real temporary Python subprocesses, derives metrics from written configurations, evaluates three candidates, and verifies rollback and completion behavior. The production entry point contains no scripted proposals or prewritten experiment results.

## Completion and Safety Stops

Normal completion uses the `agent_finish` stop reason after all plan, experiment, recovery, and report requirements pass.

Safety stop reasons include:

- `tool_call_budget_exhausted`;
- `too_many_failures`;
- `wall_time_budget_exhausted`;
- `unrecoverable_error`.

A safety stop before all completion requirements are satisfied is reported as `failed` or `incomplete_requirements`, never as a successful research result. Existing state, trajectory, and experiment artifacts remain available for diagnosis.
