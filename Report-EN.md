# Mini AutoResearch Agent: Technical Report on a Controlled Autonomous Reinforcement-Learning Experiment System

> **Report scope and evidence baseline**  
> This report analyzes the design and implementation of the Mini AutoResearch Agent and one complete experimental run. Statements about the system are based on the source code and `configs/autoresearch.yaml` in the current working tree; statements about experimental outcomes use only `runs/research_agent_003`, whose status is `completed` and whose completion requirements were verified by the harness. That historical run corresponds to repository `HEAD` `94c5f8af464f557c05ffa1f03cc503de99dff5dd`. Validation was performed on 2026-08-13. Unless explicitly labeled as an interpretation, hypothesis, or limitation, this report does not treat subjective LLM judgments as system facts.

**Contents:** [Introduction](#introduction) · [System Architecture](#system-architecture) · [Autonomous Experimentation Workflow](#autonomous-experimentation-workflow) · [Key Design Decisions](#key-design-decisions-and-safety-mechanisms) · [Experiments and Results](#experiments-and-results) · [Limitations and Future Improvements](#failure-cases-limitations-and-future-improvements) · [Conclusion](#conclusion) · [Evidence Index](#evidence-index)

## Introduction

Mini AutoResearch Agent is an autonomous research harness for small-scale machine-learning experiments. The project uses a large language model for planning, hypothesis generation, tool selection, and result interpretation, while a deterministic Python control layer governs file access, subprocess execution, metric validation, rollback, budgets, and termination. The included task performs a bounded search over Flappy Bird Q-learning hyperparameters while keeping the Q-table state space fixed. Its objective is to improve `mean_score` over 100 evaluation episodes, with `mean_reward` serving as a diagnostic and tie-breaking metric.

The central goal is not simply to have a model suggest a parameter set. Instead, the project establishes an experimental loop that executes real workloads, recovers from regressions, verifies results, and preserves a traceable record. A complete run covers goal ingestion, explicit planning, a baseline, candidate experiments, independent comparison, snapshot rollback, report generation, and completion validation. Every candidate result comes from actual training and evaluation subprocesses rather than an LLM prediction or a prewritten outcome.

The system has the following defining properties:

- **Real execution:** the LLM writes run-scoped configuration and launches actual training and evaluation through controlled tools.
- **Separation of responsibilities:** the LLM proposes actions, while the harness decides whether those actions are legal and independently validates their results.
- **Bounded autonomy:** the search space, paths, commands, experiment count, tool-call count, and runtime all have explicit limits.
- **Failure recovery:** the accepted configuration is saved before every candidate, and restored contents are verified after a rollback.
- **Complete auditability:** conversations, requests, tool arguments, metrics, logs, state transitions, and the final report are persisted.
- **Testability:** unit and end-to-end tests cover SDK message handling, path isolation, metric validation, rollback, and completion gates.

In the completed `research_agent_003` run, the system executed one baseline and five candidate experiments. Thirty-nine LLM calls corresponded to 39 tool calls, and the run ended normally with `agent_finish`. The best candidate increased `mean_score` from the baseline value of 5.20 to 14.31, an absolute gain of 9.11 and a relative gain of approximately 175.2%. This demonstrates a clear improvement under the fixed seed and evaluation protocol used by this run. Because only one seed was used and evaluation variance was high, the result does not establish cross-seed generalization or statistical significance.

## System Architecture

### Architecture Overview

The system uses a layered architecture comprising an LLM decision layer, a controlled execution layer, a task runtime layer, and an evidence layer. `main.py` parses command-line arguments and invokes `run_agent()`. The main control loop is implemented in `autoresearch/agent.py`, model transport is handled by `autoresearch/planner.py`, and every executable action is centralized in `autoresearch/tooling.py`. The task adapter is `tasks/flappy_qlearning/run.py`, while deterministic metric comparison and report validation are implemented in `autoresearch/execution.py` and `autoresearch/report.py`, respectively.

```text
Natural-language goal + runtime configuration
                    │
                    ▼
┌─────────────────────────────┐
│ LLM Agent / Planner         │  Creates plans, hypotheses, and the next tool call
└──────────────┬──────────────┘
               │ One structured tool call
               ▼
┌─────────────────────────────┐
│ Tool Harness                │  Validates paths, commands, configs, and preconditions
└───────┬─────────────┬───────┘
        │             │
        ▼             ▼
 File/snapshot     Experiment Runner
 operations             │ train → evaluate
                        ▼
                 Metrics, logs, return codes
        │             │
        └──────┬──────┘
               ▼
┌─────────────────────────────┐
│ Validator + State Machine   │  Accept/rollback, budgets, completion gates
└──────────────┬──────────────┘
               ▼
 state / messages / trajectory / report
```

Two complementary data paths run through this architecture:

1. **Interaction path:** the system prompt, original goal, recent assistant/tool turns, and authoritative state summary form the LLM's decision context.
2. **Evidence path:** configuration, subprocess state, metric files, snapshots, and structured state form the harness's authoritative evidence.

Neither path replaces the other. The LLM may interpret a tool result, but it cannot directly declare that result verified. Only facts recalculated and accepted by the harness from files and process state may update the best configuration or allow the run to finish.

### Main Components

#### LLM Agent / Planner

The LLM Agent is the system's decision-maker. It turns a natural-language research goal into an executable experiment process, but it cannot read or write files or start processes directly. At the beginning of a run, `agent.py` builds a system prompt containing the research objective, run directory, target metric, experiment count, rollback requirement, file permissions, and complete experiment protocol. The model must first submit a plan with at least five steps. It then selects one action per turn using a machine-generated authoritative state summary, recent complete tool turns, and the current observation.

The Planner invokes the configured model through an OpenAI-compatible Chat Completions API. Each request includes compact model-facing context and native function-calling schemas, with `tool_choice="required"` and `parallel_tool_calls=False`. The model must therefore choose exactly one tool per turn, creating a clear decision–execution–observation boundary. `planner.py` normalizes the response and parses the tool name, call ID, and JSON arguments. Kimi/DashScope-specific `reasoning_content` is preserved and replayed together with assistant turns retained in the recent window.

The LLM performs the following cognitive tasks:

- read the goal, runtime configuration, task code, historical metrics, and logs to establish experimental context;
- propose candidate hyperparameter configurations with a hypothesis and expected effect;
- decide, from metric evidence, whether to accept or roll back a candidate;
- reflect on tool errors or failed experiments and select a recovery action;
- update the research plan, write the final experiment report, and request completion.

This design confines open-ended reasoning to proposing actions and interpreting evidence. Path safety, command legality, metric comparison, and completion gates are enforced by deterministic Python code; model statements are never accepted as facts by default.

#### Tool

The tool layer is implemented by `ToolHarness` in `tooling.py` and is the LLM's only execution interface to the real environment. The system exposes nine native tools: `submit_plan`, `update_plan`, `list_files`, `read_file`, `write_file`, `run_command`, `evaluate_result`, `restore_snapshot`, and `finish`. The first two maintain the plan; the file tools collect or write information; the experiment tools run and evaluate candidates; and the final two handle recovery and controlled completion.

The Tool Harness does not expose unrestricted Python or shell execution. It enforces constraints before every action:

- the first tool call must be `submit_plan`, and the plan must have enough steps with unique IDs;
- reads are restricted to configured `read_roots`, and writes are confined to the current `runs/<run_id>/` directory;
- file reads are line- and character-limited, while writes are size-limited and record before/after SHA-256 hashes;
- `run_command` accepts an argument array, uses `shell=False`, and permits only the configured Python experiment script;
- the configuration path, run ID, experiment type, and timeout in the command must match current state;
- candidate configurations may change only fields and values declared in `search_space`, and may not repeat an earlier configuration.

Every tool call returns a structured `ok` status, result evidence, and `next_action`. Expected path, parameter, state, or subprocess errors are converted into tool observations and returned to the LLM instead of breaking the control loop. This allows the agent to correct real errors while preserving environmental boundaries.

#### Experiment Runner

Experiment execution uses a two-level process structure. At the outer level, `ToolHarness._tool_run_command()` invokes the configured `tasks/flappy_qlearning/run.py`. It validates the command, creates the experiment directory, applies a timeout, and captures stdout, stderr, and the return code. The baseline must exactly match the baseline configuration. Before each candidate, the system saves the last accepted configuration as a snapshot and records the candidate configuration and its changes, enabling deterministic rollback.

The inner task adapter, `tasks/flappy_qlearning/run.py`, further divides one experiment into training and evaluation stages:

1. Read and normalize the YAML configuration, then create independent train and evaluation configurations.
2. Launch the training subprocess, which produces a new Q-table and training result.
3. Launch evaluation only after successful training, with exploration disabled.
4. Aggregate both stages' return codes, durations, and metrics under a shared deadline.
5. Write `summary.json`, containing `mean_score`, `mean_reward`, training/evaluation status, and runtime data.

The Runner configures pygame with dummy video and audio drivers for headless execution. Complete output is stored in experiment log files, while only bounded log tails and structured metrics are returned to the LLM. This retains diagnostic evidence without allowing a single observation to consume excessive context.

#### State / Trace Recorder

State and trace recording are maintained jointly by `agent.py` and `ToolHarness`. They answer two different questions: what the current facts are, and how those facts were produced.

`state.json` is a serializable snapshot of the latest run state. It records the current phase, plan, baseline, current configuration, best configuration and metrics, tested configuration fingerprints, pending experiment, experiment history, failures, recovery events, and used and remaining budgets. A candidate first enters `pending_run`. It moves to immutable `history` only after `evaluate_result` and, when required, `restore_snapshot` complete. A pending experiment therefore cannot be mistaken for an accepted result.

Trace data are separated by purpose:

- `messages.jsonl` stores the complete system, user, assistant, and tool conversation;
- `trajectory.jsonl` stores tool decisions, arguments, results, errors, recovery actions, and stop events;
- `llm_calls/call_XXX/` stores each exact request and raw response; the request is written before the network call, so a stalled request remains diagnosable;
- `llm_calls/call_XXX/context.json` stores the authoritative summary, complete/request message counts, and window-compaction statistics for that call;
- `tool_calls/call_XXX_<tool>/` stores each tool's arguments and structured result;
- the baseline and iteration directories store configurations, execution metadata, metrics, stdout/stderr, and raw train/evaluation artifacts.

A thread-safe `ProgressReporter` writes terminal progress and emits periodic heartbeats during long LLM requests or experiments. Terminal output contains only bounded summaries and does not print complete prompts, file contents, or API keys. API keys are read from environment variables and are not written to run artifacts.

#### Validator

The Validator is a deterministic evidence-checking layer independent of LLM judgment. It operates during initialization, experiment execution, result decisions, rollback, and final-report generation:

- **Runtime configuration validation:** checks required sections, positive budgets, a minimum of three candidate experiments, model configuration, objective direction, metric name, experiment command, and tool allowlist.
- **Action and configuration validation:** checks path containment, command arguments, baseline equality, candidate changes against the search space, and duplicate configuration fingerprints.
- **Execution validation:** `verify_execution()` checks timeout, process return code, metric-file existence and valid JSON, successful task `status`, and the presence of a finite numeric objective metric.
- **Result validation:** `compare_metrics()` compares the candidate with the current best using the maximize/minimize direction, minimum-improvement threshold, and optional tie-breaker. The LLM may conservatively roll back an improving candidate, but it may not accept a candidate that the Validator finds declining or invalid.
- **Recovery validation:** rollback reads the pre-run snapshot, compares it with the authoritative `current_config`, copies it, reads it back for content equality, and records the restored file's SHA-256 for auditability.
- **Report validation:** checks required sections, all experiment iterations, the exact best metric, and recovery evidence, preventing the final report from contradicting persisted facts.

Tool output and LLM explanations are therefore evidence to be checked, not trusted truth. Only configurations, metrics, rollbacks, and reports accepted by the Validator can advance the state machine or complete the run.

#### Budget / Termination Controller

Budget and termination control operate at both the Agent loop and Tool Harness levels. Budgets are loaded from `configs/autoresearch.yaml` and copied into run state. The current defaults require at least three and allow at most five candidate experiments, permit 40 tool calls, allow three consecutive failures, cap total wall-clock time at 30 minutes, and limit each experiment to 300 seconds. `remaining_budget` is recalculated after every turn so that budget consumption remains explicit state.

Before each LLM call, the main loop checks tool-call count, consecutive failures, and elapsed wall-clock time. Before execution, the tool layer again checks candidate and tool budgets. It clamps the requested experiment timeout to the smaller of the per-experiment limit and remaining wall-clock time, preventing a subprocess from exceeding the global deadline. A successful tool call resets the consecutive-failure counter; failed LLM requests, invalid tool calls, and tool errors increase failure records.

The system distinguishes normal completion from safety exits:

- `agent_finish`: all plan steps are complete, the minimum experiment count is met, valid rollback evidence exists, and the final report passes validation;
- `tool_call_budget_exhausted`: the tool-call budget is exhausted, producing `incomplete_requirements`;
- `wall_time_budget_exhausted`: the wall-clock budget is exhausted, producing `incomplete_requirements`;
- `too_many_failures`: consecutive failures reach the limit, producing `failed`;
- `unrecoverable_error`: an unexpected exception terminates the run, producing `failed`.

These two levels of checks prevent the model from bypassing completion requirements by calling `finish` early, repeatedly experimenting, or requesting an excessively long process. Even when a safety limit terminates a run, existing state, trajectory, and experiment artifacts are persisted in `finally` for later diagnosis.

### Configuration and Reproducibility

One YAML file defines the model, budgets, experiment protocol, optimization objective, baseline, search space, and tool permissions. The default is `configs/autoresearch.yaml`. At the start of a run, it is copied to `runs/<run_id>/run_config.yaml`, so later audits do not depend on a global configuration that may have changed.

The main configuration settings are:

| Category | Current setting | Purpose |
|---|---|---|
| Planner | `kimi/kimi-k3`, OpenAI-compatible endpoint, `temperature=0.2` | Produces native tool-call decisions |
| Context | 4 recent complete turns, at most 24,000 characters, authoritative JSON summary on every call | Bounds history growth and exposes current facts directly |
| API authentication | `DASHSCOPE_API_KEY` environment variable | Keeps the key out of configuration and run artifacts |
| Primary objective | Maximize `mean_score`, target 5.0 | Determines whether a candidate improves on the current best |
| Tie-breaker | `mean_reward` | Used only when the primary metric does not exceed the minimum-improvement threshold |
| Experiment budget | 3–5 candidates, 40 tool calls, 30 minutes wall-clock time | Bounds autonomous search |
| Failure budget | At most 3 consecutive failures | Prevents unlimited invalid retries |
| Subprocess budget | At most 300 seconds per experiment | Bounds training and evaluation runtime |
| Evaluation protocol | 100 exploration-free episodes after training | Produces comparable candidate metrics |
| File permissions | Read `tasks/flappy_qlearning`, `configs`, and the current run; write only the current run | Isolates source code and other runs |

The baseline uses `seed=0`, 1,000 training episodes, 100 evaluation episodes, heuristic Q-table initialization, `learning_rate=0.6`, `discount_factor=0.95`, a fixed exploration rate of 0.001, and `sample_t=3`. The goal requires the state space to remain unchanged, so `dx_dy_bin_size=25` and `velocity_bin_size=2` are not search variables. The allowed search fields are:

- `training_episodes ∈ {1000, 1500, 2000}`;
- `learning_rate ∈ {0.3, 0.4, 0.6, 0.7}`;
- `epsilon_start ∈ {0, 0.001, 0.01, 0.05}`;
- `epsilon_end ∈ {0, 0.001}`;
- `epsilon_decay ∈ {0.99, 0.995, 1.0}`;
- `discount_factor ∈ {0.9, 0.95, 0.99}`;
- `sample_t ∈ {2, 3, 4}`;
- `rewards.alive ∈ {0.02, 0.05, 0.1}`.

Each run ID must name a new directory, and the program uses `exist_ok=False` to reject overwriting an earlier run. Every candidate's configuration, raw metrics, trained Q-table, logs, and execution metadata are stored in an independent directory. `fingerprint()` detects duplicate candidates by using a stable serialization of the complete configuration. Reproduction should use the same source revision, `run_config.yaml`, input Q-table, Python dependencies, and random seed, together with a new run ID:

```powershell
python -m pip install -r requirements.txt
$env:DASHSCOPE_API_KEY="<your-key>"
python main.py --config configs/autoresearch.yaml --run-id <new-run-id>
python -m unittest discover -s tests -v
```

`requirements.txt` constrains major versions of NumPy, pygame, PyYAML, the OpenAI SDK, and setuptools, but does not provide a complete lockfile. The operating system, Python minor version, and hardware are also absent from the experiment summary. The current design therefore reproduces the protocol and configuration, but cannot guarantee bitwise-identical results across environments. A public reproduction should additionally record `python --version`, platform information, a complete dependency freeze, and the source commit.

## Autonomous Experimentation Workflow

### Initialization and Context Collection

Before any model request or experiment execution, `run_agent()` performs deterministic initialization:

1. Resolve and normalize the repository root, run directory, and configuration path.
2. Load the YAML configuration and use `_validate_runtime_config()` to check required fields, budget relationships, objective direction, model provider, experiment command, and allowlist.
3. Verify that the run ID is a single directory name and create a unique directory under `runs/` with `exist_ok=False`.
4. Read the natural-language goal from `goal_file`, unless a command-line `--goal` override is supplied.
5. Create initial state containing the goal, phase, budgets, counters, completion requirements, and empty experiment history.
6. Write the runtime configuration and initial `state.json` to disk.
7. Build the system prompt, user goal, and nine tool schemas, then initialize `messages`.
8. Create a `ToolHarness` bound to the current run, configured read roots, and a single writable directory.

The system prompt defines more than the objective. It specifies the mandatory experiment protocol: submit a plan first; inspect the goal, configuration, and source files; write `current_config.yaml` exactly matching the baseline; run the baseline; create candidates only within the declared search space; inspect and accept or roll back every result; and finally produce a report containing all required sections and iteration evidence. The baseline is not an implicit Python step. The LLM must explicitly call `write_file` and `run_command`, so baseline actions enter the same audit trail as candidate actions.

Context is collected on demand. The Agent may list approved directories and read sections of the goal, configuration, source code, metrics, and logs. `read_file`, however, returns at most 500 lines and 12,000 characters by default. The model can therefore plan from actual project information without allowing one file read to consume excessive context.

### Context Management

#### Overall Strategy

The project combines a complete audit history, a bounded request window, and an authoritative state summary. Model conversation and experiment state remain separate: the Python Harness maintains authoritative structured state and validates every transition, then derives a compact JSON summary that exposes current facts directly to the LLM before each request.

```text
complete messages ───────────→ messages.jsonl (full audit)
        │
        ├─ system prompt + original goal
        ├─ 4 recent complete assistant/tool turns (≤ 24,000 chars)
state ───→ AUTHORITATIVE_STATE_SUMMARY (JSON)
        │
        └────────────────────→ model request
                                  ↓
                         request.json + context.json
```

#### Conversation and State

Every run still begins with a fixed system prompt and natural-language goal. Model responses and tool results remain append-only in the in-memory `messages` list and `messages.jsonl`, but network requests no longer replay the entire history. The Harness treats an assistant message and its following tool result or correction user message as an indivisible turn group. It retains at most the four newest groups; if their serialized form exceeds 24,000 characters, it removes the oldest groups whole. This prevents a tool result from appearing without the assistant `tool_calls` that produced it. Kimi/DashScope `reasoning_content` is replayed with each retained assistant turn.

Before every model call, the Harness derives an authoritative `schema_version=1` summary from `state`. It includes run phase, iteration progress, plan status, complete current and best configurations, best metrics, the pending candidate, last experiment result, a compact latest-tool observation, completion requirements, failures, and live remaining budgets. The latest-tool observation excludes file bodies, log tails, and large file lists while retaining outcomes, paths, hashes, metrics, next actions, and omitted-payload sizes. The model therefore still sees whether a tool succeeded even when an oversized turn is removed as a whole. The summary is appended to the request as a temporary user JSON message and states that it takes precedence when older messages conflict. This derived message is not appended to `messages.jsonl`. The LLM still cannot modify authoritative state directly; only controlled tools may trigger transitions.

#### Persistence and Traceability

Each run directory preserves the following context records:

- `messages.jsonl`: the complete LLM, user, and tool message sequence;
- `state.json`: the latest structured runtime state;
- `trajectory.jsonl`: tool calls, decisions, errors, recoveries, and next actions;
- `llm_calls/call_XXX/`: the complete request and raw response for each model call;
- `llm_calls/call_XXX/context.json`: the summary, retained/omitted turn counts, character use, and message counts for that call;
- `tool_calls/call_XXX_<tool>/`: arguments and results for each tool invocation.

`request.json` represents the compact context actually shown to the model, while `messages.jsonl` represents the complete unpruned interaction; `context.json` connects them with construction metadata. These records support replay, audit, and fault diagnosis. The project still does not provide an entry point that automatically resumes an interrupted task from `state.json` and `messages.jsonl`; calling `run_agent()` again creates a new run. Historical run `research_agent_003` predates this feature and therefore has no `context.json` files. Its experimental metrics are unaffected by the context-layer change.

#### Context Growth and Limitations

The project bounds both individual observations and cumulative recent history. A file read returns at most 500 lines and 12,000 characters by default, file lists contain at most 200 items, experiment commands return only stdout and stderr tails, and model requests retain no more than four complete recent turns totaling at most 24,000 characters. Setting `context.enabled=false` restores full-history requests for compatibility or diagnosis.

The window is measured in serialized JSON characters, not exact model tokenizer units. The fixed system prompt, original goal, tool schemas, and state summary are also outside `max_recent_chars`. The system still has no pre-request token estimate, specialized context-limit recovery, or cross-process resume capability. Longer tasks should add model-aware token budgeting, a summary-size bound, long-term semantic memory for evicted rationale, and persistent recovery.

#### Post-Implementation Comparison

To evaluate the change, the full-history run `research_agent_003` and the authoritative-summary/sliding-window run `context_control_001` were aligned at the point where Iteration 1 had been accepted. Both runs had the same baseline `mean_score` of 5.20. Token figures come from the API usage fields in each `response.json` and include successful responses only.

| Metric | Before optimization | After optimization | Change |
|---|---:|---:|---:|
| Successful LLM calls required to reach this point | 17 | 19 | +2 |
| Cumulative prompt tokens | 248,855 | 121,842 | **-51.0%** |
| Cumulative completion tokens | 5,918 | 9,515 | +60.8% |
| Cumulative total tokens | 254,773 | 131,357 | **-48.4%** |
| Iteration 1 `mean_score` | 10.47 | **18.96** | **+81.1%** |
| Iteration 1 `mean_reward` | 218.29 | **459.66** | **+110.6%** |

For the individual request that produced the Iteration 1 evaluation decision, prompt usage fell from 26,057 to 7,116 tokens, a reduction of **72.7%**. The optimized request contained only 11 messages and the four most recent complete turn groups, omitting 14 older groups. The earlier run consumed 1,012,192 total tokens across five candidates and reached a best `mean_score` of 14.31; the new run had already observed 18.96 after its first candidate and 131,357 cumulative tokens. In this run, the change therefore coincided with both lower context cost and higher early search efficiency.

This comparison is not a controlled causal A/B experiment. The two Iteration 1 candidates used different configurations, and both evaluations used only `seed=0`. The new run also required tool corrections for a duplicate plan submission, invalid parameters, and an excessive command timeout. It later stopped after Iteration 1 when DashScope returned an insufficient-balance HTTP 429 error, so it did not complete all experiments, the required rollback evidence, or the final report. The evidence establishes that context compaction substantially reduced token growth; attributing the score improvement to context management requires repeated complete runs with identical candidates and multiple random seeds. The underlying records are available in [`research_agent_003/state.json`](runs/research_agent_003/state.json), [`context_control_001/state.json`](runs/context_control_001/state.json), and the respective `llm_calls/` directories.

### Plan–Act–Observe–Validate Loop

The main loop consists of five consecutive stages. Although the LLM issues only one tool call per turn, multiple turns combine into a complete experiment. State, plan status, and tool preconditions ensure that ordering is not enforced by prompt compliance alone.

#### Planning

The first model response must call `submit_plan`. The plan must contain at least five steps with unique IDs. Every step describes expected tools and a success signal, and the plan also records risks. Later `update_plan` calls mark steps as `pending`, `in_progress`, `completed`, or `failed`, together with evidence and the next action. `finish` rejects a run with any incomplete plan step, so the plan is both working memory for the LLM and part of the completion gate.

Experiment planning follows an incremental strategy based on the current accepted configuration. `run_command` requires non-empty `hypothesis` and `expected_effect` arguments for a candidate. These values enter `pending_run` with the candidate configuration and result, and later become part of experiment history. The system verifies parameter legality but does not automatically determine whether a scientific hypothesis is sound. The report must therefore distinguish hypotheses from observed conclusions.

#### Tool Execution

`parse_tool_call()` parses each model response. If the model calls no tool, calls multiple tools, or supplies invalid JSON arguments, the Agent constructs an error observation and asks for a correction on the next turn. For a legal call, arguments are written to `tool_calls/call_XXX_<tool>/arguments.json` before `ToolHarness.execute()` dispatches the operation.

File operations compare normalized absolute paths against containment boundaries. Command execution validates the Python executable, script, `--config`, `--run-id`, and `--timeout-seconds`. Arguments containing pipes, redirects, or command separators are rejected. Experiments run through `subprocess.run(..., shell=False, capture_output=True)`, so shell syntax cannot be used to combine additional commands.

Before a candidate starts, the Harness compares `current_config.yaml` on disk with the accepted configuration. It rejects a candidate with no changes, changes outside the search space, disallowed values, or a previously tested fingerprint. Only after validation does it increment the iteration, create the experiment directory, and save a pre-run snapshot.

#### Observation

Each tool result is serialized as a `role="tool"` message and appended to the conversation. It contains a success status, structured result, error information when applicable, and a recommended `next_action`. For an experiment, observations include:

- the outer process return code and timeout status;
- the task-generated metric object;
- any problems found by independent validation;
- up to the final 2,000 characters of stdout and stderr;
- the candidate snapshot ID;
- a recommended next step to evaluate or roll back the result.

Complete stdout and stderr do not enter terminal output or the LLM context; they remain in the experiment directory. The model receives sufficient diagnostic information, auditors retain the complete evidence, and log noise and potential sensitive-content exposure are reduced.

#### Validation

After an experiment, the Harness does not trust the process's printed self-report. It reopens `summary.json` at the configured `metrics_path`. Validation requires that the process did not time out, returned zero, produced a valid JSON object, reported `status="success"`, and included a finite numeric primary metric. Any failure is preserved in `verification_errors`, and a candidate with such errors can proceed only toward rollback.

For valid metrics, `compare_metrics()` compares against the current `best_metrics`, not merely the immediately preceding experiment. The current settings are `direction=maximize` and `min_improvement=0.0`: a candidate is accepted directly only when `mean_score` is strictly higher. When the difference does not exceed the threshold, `mean_reward` is considered. Thus, `mean_reward` is a diagnostic and tie-breaking metric, not a secondary objective that may override a lower primary score.

#### Decision and State Update

The LLM calls `evaluate_result` with an `accept` or `rollback` decision, a reason, and evidence. The Harness independently calculates its own decision:

- if the LLM requests acceptance and independent validation also accepts, the candidate becomes the new `current_config` and `best_config`;
- the LLM may conservatively roll back a candidate that actually improved;
- if the LLM requests acceptance of a declining or invalid result, the tool rejects the request and requires a correction;
- after a rollback decision, no new candidate may start and the run may not finish until restoration completes.

`restore_snapshot` verifies the snapshot ID, compares the snapshot with the authoritative accepted configuration, copies it to `current_config.yaml`, and reads it back to confirm equality. The candidate then moves from `pending_run` to `history`, and the recovery event records its iteration, reason, snapshot, and SHA-256. Accepted candidates also move to `history`, but do not produce recovery records.

### State Tracking and Traceability

The system treats `state` as the authoritative state machine for one run and overwrites `state.json` after every processed tool result. Its main domains are:

| State domain | Representative fields | Purpose |
|---|---|---|
| Lifecycle | `status`, `phase`, `started_at`, `finished_at`, `stop_reason` | Describes the current phase and termination reason |
| Counts and budgets | `iteration`, `llm_call_count`, `tool_call_count`, `limits`, `remaining_budget` | Prevents infinite loops and exposes remaining capacity |
| Plan | `plan.steps[].status/evidence/next_action` | Connects task decomposition, evidence, and completion checks |
| Baseline and best | `baseline`, `current_config`, `best_config`, `best_metrics`, `best_source` | Stores accepted experimental facts |
| Candidate transaction | `pending_run`, `tested_fingerprints`, `history` | Prevents repeats and ensures candidates are evaluated before commit |
| Failure and recovery | `failures`, `last_tool_error`, `recovery_events` | Records tool errors, execution failures, and verified rollbacks |
| Completion | `requirements`, `agent_summary` | Records minimum iteration, recovery, and report-validation status |

Persistence supports three levels of reconstruction:

1. **Reconstructing the model's view:** `request.json` shows the exact compact context received by the model, `context.json` explains its summary and compaction, and `messages.jsonl` retains the full unpruned interaction.
2. **Reconstructing system actions:** `trajectory.jsonl` and tool directories answer what was called and what it returned.
3. **Reconstructing experimental facts:** configurations, metrics, execution metadata, logs, and Q-tables under the baseline and iteration directories answer what actually ran and what it produced.

The `research_agent_003` trajectory contains 39 `tool_call` events and one `stop` event. Tool usage comprised six `run_command` calls (one baseline and five candidates), five `evaluate_result` calls, two `restore_snapshot` calls, eight `read_file` calls, one `list_files` call, seven `write_file` calls, one `submit_plan` call, eight `update_plan` calls, and one `finish` call. Final state marks all three requirements as `true`, consistent with `completed / agent_finish`.

## Key Design Decisions and Safety Mechanisms

### Separation of Reasoning and Execution

The system deliberately separates nondeterministic research reasoning from deterministic environment execution. The LLM may decide what to try next and how to interpret an observation, but cannot directly access the filesystem or `subprocess`; all side effects pass through the Tool Harness. Consequently, a hallucinated, incomplete, or malformed model output remains only a request to validate and does not automatically become a system fact.

This separation also makes deterministic testing possible. Tests can replace the network model with a scripted LLM action sequence while retaining the real tools, state transitions, and subprocesses. Production and test paths share the same `run_agent()` and `ToolHarness`; only `chat_fn` is replaced. Protocol behavior can therefore be validated without mixing model randomness into control-layer tests.

### Restricted Tool Execution

The tool system uses allowlists rather than denylists. Read roots, the single write directory, and the only permitted experiment script all come from runtime configuration. Paths are resolved to normalized absolute paths before comparison. `run_command` accepts an argument array rather than a shell string and rejects control operators such as `|`, `;`, `&&`, `||`, `>`, and `<`. Together, these controls reduce path traversal, command composition, source-file overwrite, and cross-run contamination risks.

Restrictions also apply to experimental semantics. The baseline must equal the declared configuration field-for-field. A candidate must differ from the current accepted configuration, every change must belong to the search space, its full fingerprint must be new, and its run ID must match the next iteration. These rules ensure that “Iteration N” is a legal and unique candidate in the audit trail.

### Independent Result Validation

The system follows the principle that the LLM is not the Validator. A candidate process may print a success message, and the LLM may claim an improvement, but the Harness still checks the return code, timeout state, and metric file independently. Metric comparison uses structured data and fixed rules rather than natural-language conclusions. The final report must also contain required sections, every `Iteration N`, the exact best metric, and a recovery discussion when relevant.

The regression suite provides corresponding safeguards. On 2026-08-13, the following command was run under Python 3.11.14, NumPy 2.4.1, pygame 2.6.1, PyYAML 6.0.3, and OpenAI SDK 2.54.0:

```powershell
python -m unittest discover -s tests -v
```

All twelve tests passed. Coverage includes native tool-call parsing and Kimi reasoning replay, context-configuration validation, authoritative state summaries, atomic assistant/tool windows, large-payload compaction for the latest tool observation, disabled-mode compatibility, OpenAI SDK arguments, progress heartbeats and content non-disclosure, minimum runtime requirements, independent execution and metric comparison, report evidence validation, rejection of out-of-run writes and early completion, and an end-to-end state-machine test with a baseline, three candidates, a real rollback, and compaction audit records. These results support correct operation of the control and context logic, but do not prove coverage of every operating system, model provider, or long-duration failure mode.

### Rollback and Recovery Mechanism

Candidate execution is modeled as a transaction. Before execution, the Harness saves the accepted configuration as `snapshots/iteration_NN_before.yaml`, and the resulting candidate enters `pending_run`. If performance declines, execution is invalid, or the LLM chooses rollback, the system cannot proceed directly to another experiment; it requires an explicit `restore_snapshot` call.

Restoration checks four conditions: the pending candidate has been evaluated, its decision is rollback, the requested snapshot ID matches pending state, and snapshot contents match the authoritative accepted configuration in memory. After copying, the YAML is read back and compared again, and a SHA-256 hash is recorded. Only then does the candidate enter history and `pending_run` become empty.

Normal completion may additionally require genuine recovery evidence. By default, `evidence.require_recovery_event=true`, and a recovery event qualifies only when the independent Validator also determined that the candidate required rollback. An ordinary tool-argument correction may be recorded as a recovery event, but it cannot satisfy the research requirement. This prevents the agent from manufacturing failure-and-recovery evidence by intentionally issuing an invalid command.

### Termination and Budget Control

Run completion is not a free-form model response; it is a controlled `finish` tool call. Normal completion requires no `pending_run`, all plan steps completed, the minimum candidate count reached, qualifying rollback evidence present, and the final report validated. A missing condition is returned as a tool error so the LLM may correct it within the remaining budget.

Safety termination is strictly separated from normal completion. Exhausting tool calls or wall-clock time produces `incomplete_requirements`; reaching the consecutive-failure limit or encountering an uncaught exception produces `failed`. On every exit path, `finally` updates timestamps, duration, remaining budgets, and the stop event before writing `state.json`. A stopped process is therefore never mislabeled as successful research.

The current `agent_steps` field mirrors `llm_call_count` and has no independent `max_agent_steps` configuration. Actual termination limits are tool calls, candidate count, consecutive failures, wall-clock time, and per-experiment duration. Public descriptions should not present `agent_steps` as a separate enforced budget.

## Experiments and Results

### Experimental Setup

The formal results come from `runs/research_agent_003`. The run started at 2026-08-12 21:40:38 (UTC+8) and ended at 22:02:58, for a total duration of 1,339.79 seconds (approximately 22 minutes 20 seconds). Its final status was `completed` and its stop reason was `agent_finish`. The run made 39 LLM calls and 39 tool calls, and completed one baseline plus five candidates. `state.json` reports `total_failures=0`. Here, the failure counter refers to LLM, tool, or execution-protocol errors; it does not count candidates that executed successfully but regressed in performance.

The task uses discrete Q-learning. The input Q-table has shape `(30, 100, 100, 2)` and data type `float64`. State consists of discretized horizontal distance to the next pipe (`dx`), vertical displacement from the pipe-gap center (`dy`), and vertical velocity (`vy`); the final dimension represents two actions, no flap and flap. Training applies:

```text
Q(s_t, a_t) ← (1 - α)Q(s_t, a_t)
              + α[r_t + γ max_a Q(s_{t+1}, a)]
```

The learning rate decays according to `max(learning_rate_min, learning_rate × learning_rate_decay^counter)`. Training uses epsilon-greedy action selection, while evaluation fixes epsilon at zero. Training starts with heuristic Q-value initialization, and evaluation loads the `q_table.npy` produced by that training run. Pygame uses dummy video and audio drivers, rendering is disabled, and both training and evaluation run at an FPS setting of 1500.

Every experiment uses `seed=0` and 100 evaluation episodes. The baseline trains for 1,000 episodes; candidates may choose 1,000, 1,500, or 2,000 episodes within the search space. State discretization remains fixed at `dx_dy_bin_size=25` and `velocity_bin_size=2`, satisfying the requirement not to alter the Q-table state space.

Metrics reported here come directly from each run's `summary.json` and were cross-checked against the baseline and history in final `state.json`. Because the task adapter currently assigns `mean_score` to the `min_score` field, this report does not treat `min_score` as a valid experimental metric.

### Baseline and Iterative Experiments

The baseline uses 1,000 training episodes, `learning_rate=0.6`, `discount_factor=0.95`, fixed `epsilon=0.001`, `sample_t=3`, and `rewards.alive=0.05`. It achieved `mean_score=5.20`, `std_score=4.67`, `max_score=21`, and `mean_reward=65.7870`. Because it already slightly exceeded the target of 5.0, subsequent work aimed to improve further rather than merely reach the target.

The complete candidate results are shown below. A configuration change is relative to the current best immediately before that iteration, not always to the original baseline.

| Stage | Change from the current best | `mean_score` | `std_score` | `max_score` | `mean_reward` | Training time (s) | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline | Declared baseline configuration | 5.20 | 4.67 | 21 | 65.7870 | 17.67 | Establish baseline |
| Iteration 1 | `training_episodes: 1000→2000`; `discount_factor: 0.95→0.99` | 10.47 | 8.24 | 44 | 218.2885 | 43.11 | Accept |
| Iteration 2 | `sample_t: 3→2` | 10.96 | 10.79 | 54 | 196.8880 | 31.71 | Accept |
| Iteration 3 | `rewards.alive: 0.05→0.1` | 3.92 | 3.45 | 13 | 43.2380 | 31.89 | Rollback |
| Iteration 4 | `epsilon_start: 0.001→0.01`; `epsilon_end: 0.001→0`; `epsilon_decay: 1→0.995` | 6.71 | 5.25 | 24 | 85.8205 | 30.06 | Rollback |
| Iteration 5 | `learning_rate: 0.6→0.4` | **14.31** | 16.39 | **92** | **287.3115** | 44.31 | Accept |

Iteration-by-iteration analysis follows:

- **Iteration 1:** increasing both training episodes and the discount factor raised `mean_score` from 5.20 to 10.47, so the Harness accepted the candidate. Because two factors changed together, this run cannot isolate whether the improvement came from training duration, discount factor, or their interaction.
- **Iteration 2:** reducing `sample_t` from 3 to 2 on the accepted configuration produced a small increase in `mean_score` to 10.96, so the deterministic rule accepted it. However, `mean_reward` fell from 218.2885 to 196.8880 and score standard deviation was 10.79. The gain of 0.49 is smaller than the observed scale of variation and should not be interpreted as a demonstrated robust improvement.
- **Iteration 3:** increasing the survival reward to 0.1 reduced `mean_score` to 3.92, below the current best of 10.96. The system rolled back to the Iteration 2 configuration. The data establish that this setting performed poorly under the current protocol, but do not establish a particular behavioral cause.
- **Iteration 4:** starting again from the restored Iteration 2 configuration, the candidate introduced exploration that decayed from 0.01 to zero. Its `mean_score` was 6.71, still below 10.96, so the system rolled back again. The result shows that this joint exploration schedule did not improve the metric within the current training budget; it cannot isolate the contributions of start value, end value, and decay rate.
- **Iteration 5:** starting from the restored Iteration 2 configuration, only the initial learning rate changed, from 0.6 to 0.4. `mean_score` increased to 14.31, and the Harness accepted it as the final best. Although both mean score and mean reward improved, `std_score` also increased from 10.79 to 16.39, so the result does not support a claim of lower variance or a more stable policy.

Of five candidates, three were accepted and two were rolled back. All six experiment processes, including the baseline, reported `status="success"`; every train and evaluation return code was zero, and no verification errors were recorded. The two failures were performance regressions from legal candidates, not command errors or process failures.

### Best Result

The final best result is Iteration 5, whose key configuration is:

| Parameter | Best value |
|---|---:|
| `training_episodes` | 2000 |
| `evaluation_episodes` | 100 |
| `learning_rate` | 0.4 |
| `learning_rate_min` | 0.1 |
| `learning_rate_decay` | 0.9995 |
| `discount_factor` | 0.99 |
| `epsilon_start / end / decay` | 0.001 / 0.001 / 1.0 |
| `sample_t` | 2 |
| `rewards.alive` | 0.05 |
| `q_init` | `heuristic` |
| `seed` | 0 |

Relative to the baseline:

- `mean_score`: 5.20 → **14.31**, an absolute gain of **9.11**, a relative gain of **175.2%**, and **2.752×** the baseline;
- relative to the configured target of 5.0: **9.31** higher and **2.862×** the target;
- `mean_reward`: 65.7870 → **287.3115**, an absolute gain of **221.5245**;
- `max_score`: 21 → **92**;
- final `std_score=16.39`, indicating substantial variation among evaluation episodes.

It is therefore justified to state that the best observed candidate clearly outperformed the baseline under this fixed seed, fixed state space, and 100-episode evaluation protocol. The evidence does not justify claiming an equivalent gain across multiple random seeds, machines, or initialization conditions.

### Execution Trace Example

The key events in `research_agent_003` illustrate how model decisions become controlled state transitions:

```text
call 01  submit_plan                         → plan established
call 02–10  read/list files                 → collect goal, config, and task code
call 12  write current_config.yaml          → write baseline
call 13  run baseline                       → mean_score 5.20
call 15–17  write/run/evaluate Iteration 1  → 10.47, accept
call 19–21  write/run/evaluate Iteration 2  → 10.96, accept
call 23–25  write/run/evaluate Iteration 3  → 3.92, rollback
call 26  restore iteration_03_before.yaml   → verified=true
call 28–30  write/run/evaluate Iteration 4  → 6.71, rollback
call 31  restore iteration_04_before.yaml   → verified=true
call 33–35  write/run/evaluate Iteration 5  → 14.31, accept
call 37  write final report                 → report persisted
call 39  finish                             → completed / agent_finish
```

Both recoveries recorded the snapshot-content hash `a8ce474372b7b6bbc7b3eab8838f625ecfd27f1a14dad16d573ac5cf0e673f4c`, corresponding to the accepted Iteration 2 configuration. Final state shows that all five candidates entered history, `pending_run=null`, the best source is `iteration_05`, and all three completion requirements—`min_iterations_met`, `recovery_demonstrated`, and `report_verified`—are `true`.

## Failure Cases, Limitations, and Future Improvements

### Failure Cases and Recovery

#### Observed Performance Regressions

The completed run contained no LLM request failure, tool-protocol error, subprocess exception, or metric-validation error; `total_failures=0`. The observed failures were two candidates that executed successfully but reduced performance:

- **Iteration 3:** with `rewards.alive=0.1`, `mean_score` fell from the current best of 10.96 to 3.92. `evaluate_result` returned rollback, after which `iteration_03_before.yaml` was restored.
- **Iteration 4:** changing the epsilon start, end, and decay together produced `mean_score=6.71`, still below 10.96. The system restored `iteration_04_before.yaml`.

Both `restore_snapshot` calls returned `restored=true, verified=true` and recovered the same accepted Iteration 2 configuration. Iteration 5 consequently started from a clean restored state rather than building on a regressed candidate. This demonstrates that snapshot recovery functioned in real performance-regression scenarios during this run.

#### Handled Protocol and Execution Failures

The code handles the following failure paths, some of which are covered by automated tests but did not occur in `research_agent_003`:

| Failure type | System response | Current test coverage |
|---|---|---|
| Model calls no tool or multiple tools | Record failure and return a correction message | Native call protocol covered |
| Invalid tool JSON, path, or argument | Return `ok=false` and increment consecutive failures | Out-of-run write covered |
| Baseline mismatch or out-of-space/duplicate candidate | Reject experiment without starting a subprocess | Harness logic present; more combinational tests are possible |
| Subprocess timeout, nonzero return code, or missing metrics | Record execution failure and require evaluation and rollback | Nonzero return-code validation covered |
| LLM accepts a declining result | `evaluate_result` rejects acceptance | Independent comparison covered |
| Premature `finish` | Return missing conditions and keep the run active | Covered |
| Network/API error | Record LLM failure and retry within the failure budget | SDK error conversion exists; no real network-failure integration test |
| Uncaught exception | Mark `failed / unrecoverable_error` and persist state in `finally` | Fault-injection test missing |

A successful tool call resets the consecutive-failure counter and may record a previous correction as `tool_correction`, which does not satisfy the research recovery requirement. Only a verified rollback of a genuinely invalid or declining candidate can satisfy `recovery_demonstrated`.

### Current Limitations

#### Experimental Validity

- **Single random seed:** every experiment uses `seed=0`, so the run cannot estimate cross-seed mean, confidence intervals, or failure probability.
- **High evaluation variance:** the best result has `std_score=16.39`, greater than `mean_score=14.31`. Although 100 episodes characterize this evaluation, they are insufficient to support a robust ranking of small differences.
- **Adaptive and incomplete ablation design:** Iteration 1 changes training episodes and discount factor together, while Iteration 4 changes three epsilon parameters together, preventing single-factor causal attribution.
- **Overly permissive acceptance threshold:** `min_improvement=0.0` accepts any strictly positive gain, even when it is much smaller than evaluation noise, as in Iteration 2's +0.49.
- **Limited search budget:** only five discrete candidates are allowed. The run does not repeat a configuration or systematically explore parameter interactions.
- **Baseline already above target:** the target of 5.0 is not challenging for the baseline. This run contributes further optimization rather than demonstrating recovery from a below-target starting point.

#### Metrics and Reporting

- `summarize()` in `tasks/flappy_qlearning/run.py` currently writes `mean_score` into `min_score`, so the produced `min_score` values are invalid.
- `max_score` is sensitive to outlier episodes and cannot replace a mean or distributional statistics. The system currently reports no median, quantiles, or confidence interval.
- `validate_report()` checks required section names, Iteration references, the exact best metric, and recovery keywords, but cannot validate every parameter, percentage, or causal explanation in the report.
- LLM-generated interpretations may exceed the evidence. Inferring a particular policy behavior from one hyperparameter change requires trajectory analysis or controlled replication, which the current Harness does not perform automatically.

#### Context and Durability

- The authoritative state summary and character-budgeted sliding window are implemented, but 24,000 characters are only a proxy for token usage, and the summary itself has no independent size limit.
- The window retains at most four recent turns. Earlier source observations and research rationale may leave the request; factual state is preserved by the summary, but it is not a substitute for long-term semantic memory.
- `messages.jsonl`, `state.json`, and per-call requests are auditable, but `run_agent()` cannot resume an interrupted run from these records.
- `state.json` is overwritten normally rather than written to a temporary file and atomically replaced, so an extreme power loss or process interruption could leave a partial write.
- Run records preserve complete prompts, model responses, and tool observations. Although API keys are not persisted, a run directory still requires content and privacy review before public distribution.

#### Reproducibility and Scope

- Dependency declarations specify ranges rather than a lockfile or container image. The completed run also lacks a manifest containing exact Python, dependency, OS, CPU, and GPU information.
- Absolute training and evaluation times depend on hardware and system load and describe only this run.
- The system currently supports only OpenAI-compatible Chat Completions providers and includes Kimi-specific `reasoning_content` compatibility.
- Tool restrictions are application-level allowlists, not an operating-system container sandbox. The approved experiment script still executes with the permissions of the host Python process.
- Conclusions apply only to the included discrete Flappy Bird Q-learning task and cannot be directly generalized to deep learning, large-scale parallel experimentation, or high-risk production systems.

### Future Improvements

The following work is recommended in priority order:

1. **Correct metric aggregation.** Fix `min_score`; add median, P25/P75, full episode scores or verifiable summaries; and add consistency tests for every numeric output field.
2. **Strengthen experiment decisions.** Run every candidate across multiple random seeds, compare cross-seed means and confidence intervals, and set `min_improvement` above the noise floor to avoid accepting statistically indistinguishable gains.
3. **Improve experimental design.** Split multi-parameter candidates into ablations or adopt budget-constrained Bayesian optimization or successive halving while retaining independent Harness validation.
4. **Capture the complete environment.** Record source commit, dirty-worktree status, exact Python and dependency versions, platform, and hardware for every run; provide a lockfile or container definition.
5. **Implement a resumable state machine.** Save state with atomic replacement, create a consistent checkpoint per turn, and resume from `state.json + messages.jsonl` after verifying the last tool call, `pending_run`, and on-disk artifacts.
6. **Deepen long-context management.** Add model-tokenizer estimates, a summary-size cap, and context-length error recovery on top of the existing authoritative summary and character window; compress evicted research rationale into long-term semantic memory with file references.
7. **Strengthen report validation.** Generate result tables and key figures directly from `state.json`, or compare machine-readable report front matter field-by-field with state; reserve natural language for interpretation rather than manual transcription of core metrics.
8. **Increase isolation and resource control.** Run experiments in a container or low-privilege process, impose CPU, memory, disk, and process-tree limits, and scan public artifacts automatically for secrets and private information.
9. **Expand fault testing.** Add fault-injection cases for timeout, corrupted metrics, missing snapshots, interrupted state writes, consecutive API failures, context overflow, and cross-platform paths.
10. **Improve run governance.** Add explicit `aborted` and `stale` statuses and cleanup policies for incomplete runs, and maintain a summary index that distinguishes completed, incomplete, and failed runs.

## Conclusion

Mini AutoResearch Agent demonstrates an autonomous experiment design with clear boundaries: the LLM performs planning and research judgment, while the Python Harness performs real execution, constraint enforcement, verification, and evidence persistence. Its value comes not from granting the model unlimited execution privileges, but from embedding open-ended model reasoning inside a deterministic state machine that can reject actions, recover from regressions, and preserve an audit trail.

The only run that completed and passed every completion requirement, `research_agent_003`, increased Flappy Bird Q-learning `mean_score` from 5.20 to 14.31 under its fixed seed and evaluation protocol. It also performed verified snapshot rollback for two legal but regressing candidates. All twelve automated tests passed, supporting the core control logic and the new authoritative-summary/sliding-window behavior. At the same time, a single seed, high variance, a metric-aggregation defect, lack of exact token budgeting and resume support, and incomplete environment locking limit the strength of the conclusions.

The current system therefore qualifies as a small-scale, controlled, and auditable autonomous experimentation prototype. Before it is used for longer, more expensive, or public benchmark studies, metric correctness, multi-seed statistical validation, environment capture, atomic recovery, and stronger isolation should be addressed. The best result in this report should be understood as the best observed result from one complete run, not as an unqualified statement of algorithmic performance.

## Evidence Index

The following locations provide the primary evidence used in this report:

- System entry point and control loop: [main.py](main.py), [autoresearch/agent.py](autoresearch/agent.py)
- Model transport and tool parsing: [autoresearch/planner.py](autoresearch/planner.py)
- Tools, state transitions, experiment validation, and rollback: [autoresearch/tooling.py](autoresearch/tooling.py)
- Metric validation and comparison: [autoresearch/execution.py](autoresearch/execution.py)
- Final-report validation: [autoresearch/report.py](autoresearch/report.py)
- Flappy Bird train/evaluation adapter: [tasks/flappy_qlearning/run.py](tasks/flappy_qlearning/run.py)
- Q-learning implementation: [tasks/flappy_qlearning/src/flappyq.py](tasks/flappy_qlearning/src/flappyq.py)
- Default experiment protocol: [configs/autoresearch.yaml](configs/autoresearch.yaml)
- Completed run state: [runs/research_agent_003/state.json](runs/research_agent_003/state.json)
- Complete tool trajectory: [runs/research_agent_003/trajectory.jsonl](runs/research_agent_003/trajectory.jsonl)
- Baseline and candidate metrics: [Baseline](runs/research_agent_003/baseline/metrics.json), [Iteration 1](runs/research_agent_003/iteration_01/metrics.json), [Iteration 2](runs/research_agent_003/iteration_02/metrics.json), [Iteration 3](runs/research_agent_003/iteration_03/metrics.json), [Iteration 4](runs/research_agent_003/iteration_04/metrics.json), [Iteration 5](runs/research_agent_003/iteration_05/metrics.json)
- Automated tests: [tests/test_core.py](tests/test_core.py)
