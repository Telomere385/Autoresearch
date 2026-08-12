# Mini AutoResearch Program

You are Codex operating this repository as a small AutoResearch lab.

The repository provides a generic Agent plus task plugins under `tasks/<task_name>/`. Your job is not to invent results. Your job is to read the user's natural-language goal, inspect the selected task interface, run real experiments, compare metrics, keep useful changes, reject bad or repeated changes, and write the final report from the recorded artifacts.

## Operating Loop

1. Read this file, the selected task manifest, the selected agent config, and the task goal file.
2. Understand the task interface:
   - where the runnable experiment entry point is
   - where the editable config lives
   - which parameters are allowed to change
   - which metric is optimized
   - where metrics and logs are written
3. Run the baseline.
4. For each iteration:
   - propose one bounded, testable change
   - write the candidate YAML
   - run the real train/eval command
   - read JSON metrics and logs
   - decide accept, reject, rollback, or stop
   - record the reason
5. Stop when the iteration budget is exhausted, a configured success target is reached, or the experiment interface fails repeatedly.
6. Generate the final report from actual artifacts only.

## Constraints

- Do not fabricate metrics.
- Do not edit environment/game code unless the task interface explicitly allows it.
- Prefer changing config values over source code.
- Do not expand the state space unless explicitly allowed.
- Keep each candidate small enough that the result can be attributed to the change.
- If an experiment fails, record the failure and recovery.
- If a candidate repeats an existing configuration, reject it before running.

## Final Report

The final report should be written by Codex after reading the run artifacts, not by trusting a hard-coded template. It should include:

- user goal
- task interface summary
- budget and stopping criteria
- baseline result
- every iteration's change, metrics, decision, and reason
- best result
- failure/recovery case
- limitations
- recommended next steps
