# Final Report Prompt

Use this prompt with Codex after an Agent run finishes.

Read:

- `docs/PROGRAM.md`
- `tasks/<task_name>/task.yaml`
- `runs/<run_id>/goal.md`
- `runs/<run_id>/state.json`
- `runs/<run_id>/decisions.jsonl`
- `runs/<run_id>/planner_calls.jsonl`
- `runs/<run_id>/tool_calls.jsonl`
- `runs/<run_id>/errors.jsonl`
- every `runs/<run_id>/iteration_*/reflection.json`
- every relevant `summary.json`

Write `reports/<run_id>/codex_report.md`.

The report must be based only on recorded artifacts. Do not invent metrics or decisions.

Required sections:

1. User goal
2. Task interface
3. Budget and stopping criteria
4. Baseline result
5. Iteration table
6. Best result
7. Failure and recovery
8. Limitations
9. Recommended next experiments

Keep the report concise and concrete.
