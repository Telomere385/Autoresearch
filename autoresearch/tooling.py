"""Tool schemas and guarded implementations exposed to the research planner."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .execution import compare_metrics, fingerprint, load_yaml, verify_execution, write_json
from .progress import ProgressReporter
from .report import validate_report


def tool_schemas():
    """Tools exposed to the LLM through the native function-calling protocol."""
    return [
        _tool("submit_plan", "Submit the multi-step research plan before doing any work.", {
            "goal_summary": _string(),
            "steps": {"type": "array", "minItems": 5, "items": {
                "type": "object",
                "properties": {
                    "id": _string(), "description": _string(),
                    "expected_tools": {"type": "array", "items": _string()},
                    "success_signal": _string(),
                },
                "required": ["id", "description", "expected_tools", "success_signal"],
                "additionalProperties": False,
            }},
            "risks": {"type": "array", "items": _string()},
        }, ["goal_summary", "steps", "risks"]),
        _tool("update_plan", "Update one plan step after observing tool evidence.", {
            "step_id": _string(),
            "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "failed"]},
            "evidence": _string(),
            "next_action": _string(),
        }, ["step_id", "status", "evidence", "next_action"]),
        _tool("list_files", "List files under an allowed workspace directory.", {
            "path": _string(), "pattern": _string(),
            "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
        }, ["path", "pattern", "max_results"]),
        _tool("read_file", "Read a text file from an allowed workspace path.", {
            "path": _string(),
            "start_line": {"type": "integer", "minimum": 1},
            "max_lines": {"type": "integer", "minimum": 1, "maximum": 1000},
        }, ["path", "start_line", "max_lines"]),
        _tool("write_file", "Write a UTF-8 file inside this run directory.", {
            "path": _string(), "content": _string(),
        }, ["path", "content"]),
        _tool("run_command", "Run the configured Python experiment through a real subprocess.", {
            "argv": {"type": "array", "minItems": 2, "items": _string()},
            "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
            "experiment_kind": {"type": "string", "enum": ["baseline", "candidate"]},
            "hypothesis": _string(),
            "expected_effect": _string(),
        }, ["argv", "timeout_seconds", "experiment_kind", "hypothesis", "expected_effect"]),
        _tool("evaluate_result", "Judge the pending candidate after inspecting its metrics and logs.", {
            "iteration": {"type": "integer", "minimum": 1},
            "decision": {"type": "string", "enum": ["accept", "rollback"]},
            "reason": _string(), "evidence": _string(),
        }, ["iteration", "decision", "reason", "evidence"]),
        _tool("restore_snapshot", "Restore the pending candidate's verified pre-experiment snapshot.", {
            "snapshot_id": _string(), "reason": _string(),
        }, ["snapshot_id", "reason"]),
        _tool("finish", "Finish only after the report and all research requirements are complete.", {
            "summary": _string(), "report_path": _string(),
        }, ["summary", "report_path"]),
    ]


class ToolHarness:
    """Execute planner-requested tools within run-scoped safety boundaries.

    The harness owns deterministic validation and state transitions. The LLM
    proposes actions and interpretations, but cannot bypass path, command,
    budget, metric, rollback, or completion checks enforced here.
    """

    def __init__(self, root, run_dir, config, state, reporter=None):
        """Bind tool execution to one workspace root and one writable run dir."""
        self.root = Path(root).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.config = config
        self.state = state
        self.reporter = reporter or ProgressReporter("quiet")
        self.current_path = self.run_dir / "current_config.yaml"
        configured_write_root = str(config.get("tools", {}).get("write_root", f"runs/{state['run_id']}"))
        configured_write_root = configured_write_root.format(run_id=state["run_id"])
        write_path = Path(configured_write_root)
        self.write_root = (
            (self.root / write_path).resolve() if not write_path.is_absolute() else write_path.resolve()
        )
        if self.write_root != self.run_dir:
            raise ValueError("tools.write_root must resolve to this run's directory")
        self.read_roots = self._read_roots()

    def execute(self, name, arguments, call_dir):
        """Dispatch one tool call and convert expected failures to observations."""
        call_dir = Path(call_dir)
        try:
            if not self.state.get("plan") and name != "submit_plan":
                raise ToolError("submit_plan must be the first tool call")
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                raise ToolError(f"Unknown tool: {name}")
            result = handler(arguments, call_dir)
            result = {"ok": True, **result}
        except (ToolError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            result = {"ok": False, "error": str(exc), "next_action": "reflect and call a corrected tool"}
            self.state["consecutive_failures"] += 1
            self.state["total_failures"] += 1
            self.state["failures"].append({
                "tool_call": self.state["tool_call_count"],
                "type": "tool_error",
                "tool": name,
                "message": str(exc),
                "recovery_action": "return error to the LLM for correction",
            })
            self.state["last_tool_error"] = {"tool": name, "message": str(exc)}
            self.reporter.emit("ERROR", f"{name}: {exc}")
        else:
            prior_error = self.state.pop("last_tool_error", None)
            if prior_error:
                self.state["recovery_events"].append({
                    "type": "tool_correction",
                    "failed_tool": prior_error["tool"],
                    "error": prior_error["message"],
                    "recovered_by": name,
                    "qualifies": False,
                })
            self.state["consecutive_failures"] = 0
        return result

    def _tool_submit_plan(self, args, _call_dir):
        """Validate and persist the LLM's initial multi-step research plan."""
        steps = args.get("steps")
        if self.state.get("plan"):
            raise ToolError("A plan has already been submitted; use update_plan")
        if not isinstance(steps, list) or len(steps) < 5:
            raise ToolError("The plan must contain at least five executable steps")
        ids = [str(step.get("id", "")) for step in steps]
        if any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ToolError("Every plan step needs a unique non-empty id")
        plan = {
            "goal_summary": _required_text(args, "goal_summary"),
            "steps": [
                {
                    "id": str(step["id"]),
                    "description": _required_text(step, "description"),
                    "expected_tools": list(step.get("expected_tools") or []),
                    "success_signal": _required_text(step, "success_signal"),
                    "status": "pending",
                    "evidence": "",
                    "next_action": "",
                }
                for step in steps
            ],
            "risks": list(args.get("risks") or []),
        }
        self.state["plan"] = plan
        self.state["phase"] = "inspection"
        write_json(self.run_dir / "plan.json", plan)
        self.reporter.emit("PLAN", f"submitted {len(steps)} steps")
        return {"plan_path": self._display(self.run_dir / "plan.json"), "step_count": len(steps)}

    def _tool_update_plan(self, args, _call_dir):
        """Update one plan step with evidence and an explicit next action."""
        step_id = str(args.get("step_id", ""))
        for step in self.state["plan"]["steps"]:
            if step["id"] == step_id:
                step["status"] = args["status"]
                step["evidence"] = _required_text(args, "evidence")
                step["next_action"] = _required_text(args, "next_action")
                write_json(self.run_dir / "plan.json", self.state["plan"])
                self.reporter.emit("PLAN", f"step {step_id} -> {step['status']}: {step['next_action']}")
                return {"updated_step": step_id, "status": step["status"]}
        raise ToolError(f"Unknown plan step: {step_id}")

    def _tool_list_files(self, args, _call_dir):
        """List bounded file metadata under an approved read root."""
        directory = self._resolve_read(args["path"])
        if not directory.is_dir():
            raise ToolError(f"Not a directory: {args['path']}")
        pattern = str(args.get("pattern") or "*")
        limit = min(int(args.get("max_results", 100)), 200)
        files = []
        for path in directory.rglob(pattern):
            if path.is_file() and self._is_readable(path):
                files.append({"path": self._display(path), "bytes": path.stat().st_size})
                if len(files) >= limit:
                    break
        self.reporter.emit("FILE", f"list {self._display(directory)} -> {len(files)} files")
        return {"files": files, "truncated": len(files) >= limit}

    def _tool_read_file(self, args, _call_dir):
        """Read a bounded UTF-8 slice without crossing approved read roots."""
        path = self._resolve_read(args["path"])
        if not path.is_file():
            raise ToolError(f"Not a file: {args['path']}")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ToolError(f"File is not UTF-8 text: {args['path']}") from exc
        start = int(args.get("start_line", 1))
        configured = int(self.config.get("tools", {}).get("max_read_lines", 500))
        maximum = min(int(args.get("max_lines", configured)), configured, 1000)
        selected = lines[start - 1:start - 1 + maximum]
        content = "\n".join(selected)
        char_limit = int(self.config.get("tools", {}).get("max_output_chars", 12000))
        truncated = len(content) > char_limit or start - 1 + maximum < len(lines)
        self.reporter.emit(
            "FILE", f"read {self._display(path)} lines {start}-{start + len(selected) - 1}"
        )
        return {
            "path": self._display(path),
            "start_line": start,
            "end_line": start + len(selected) - 1,
            "total_lines": len(lines),
            "content": content[:char_limit],
            "truncated": truncated,
        }

    def _tool_write_file(self, args, _call_dir):
        """Write a size-bounded UTF-8 artifact inside the current run only."""
        path = self._resolve_write(args["path"])
        content = args.get("content")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        if len(content.encode("utf-8")) > 250_000:
            raise ToolError("write_file content exceeds 250000 bytes")
        before = _sha256(path) if path.exists() else None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.reporter.emit("FILE", f"wrote {self._display(path)} ({path.stat().st_size} bytes)")
        return {
            "path": self._display(path), "bytes": path.stat().st_size,
            "sha256_before": before, "sha256_after": _sha256(path),
        }

    def _tool_run_command(self, args, call_dir):
        """Run one validated baseline or candidate experiment subprocess.

        Candidate runs receive a pre-experiment snapshot and remain pending
        until the planner submits an evaluation and, when needed, a rollback.
        """
        if self.state.get("pending_run"):
            raise ToolError("Evaluate and, if required, restore the pending experiment first")
        kind = args.get("experiment_kind")
        baseline = self.state.get("baseline") or {}
        if kind == "baseline" and baseline.get("status") == "success":
            raise ToolError("A successful baseline already exists")
        if kind == "candidate":
            if not self.state.get("baseline") or self.state["baseline"].get("status") != "success":
                raise ToolError("A successful baseline is required before candidate experiments")
            if self.state["iteration"] >= self.state["limits"]["max_iterations"]:
                raise ToolError("The candidate experiment budget is exhausted")

        argv, experiment_id = self._validate_command(args, kind)
        candidate_config = load_yaml(self.current_path)
        snapshot_id = None
        artifact_dir = self.run_dir / "baseline"
        if kind == "baseline":
            if candidate_config != self.config["baseline"]:
                raise ToolError("The baseline command must use the configured baseline unchanged")
        else:
            changes = _config_changes(self.state["current_config"], candidate_config)
            if not changes:
                raise ToolError("Candidate configuration is identical to the accepted configuration")
            errors = _validate_changes(changes, self.config["search_space"])
            if errors:
                raise ToolError("; ".join(errors))
            candidate_fingerprint = fingerprint(candidate_config)
            if candidate_fingerprint in self.state["tested_fingerprints"]:
                raise ToolError("Candidate repeats a previously tested configuration")
            self.state["iteration"] += 1
            iteration = self.state["iteration"]
            artifact_dir = self.run_dir / f"iteration_{iteration:02d}"
            artifact_dir.mkdir(parents=True, exist_ok=False)
            # Snapshot the last accepted configuration before executing an
            # untrusted candidate so rollback is deterministic and verifiable.
            snapshot_id = f"iteration_{iteration:02d}_before.yaml"
            snapshot_path = self.run_dir / "snapshots" / snapshot_id
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text(
                _yaml_text(self.state["current_config"]), encoding="utf-8"
            )
            shutil.copy2(self.current_path, artifact_dir / "candidate_config.yaml")
            shutil.copy2(snapshot_path, artifact_dir / "before_config.yaml")
            self.state["tested_fingerprints"].append(candidate_fingerprint)

        artifact_dir.mkdir(parents=True, exist_ok=True)
        timeout = self._bounded_timeout(float(args["timeout_seconds"]))
        label = "baseline" if kind == "baseline" else f"Iteration {self.state['iteration']}"
        self.reporter.emit("EXP", f"{label} started, timeout={timeout:g}s")
        started = time.time()
        timed_out = False
        try:
            with self.reporter.waiting("EXP", f"{label} running"):
                completed = subprocess.run(
                    argv, cwd=str(self.root), text=True, capture_output=True, timeout=timeout + 5,
                )
            returncode, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = None
            stdout = _decode_timeout(exc.stdout)
            stderr = _decode_timeout(exc.stderr) + f"\nTimed out after {timeout} seconds.\n"
        duration = time.time() - started
        (artifact_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (artifact_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        metrics_path = self.root / self.config["experiment"]["metrics_path"].format(
            experiment_id=experiment_id
        )
        metrics, errors = verify_execution(
            returncode, timed_out, metrics_path, self.config["objective"]
        )
        if metrics is not None:
            write_json(artifact_dir / "metrics.json", metrics)
        execution = {
            "argv": argv, "returncode": returncode, "timed_out": timed_out,
            "duration_seconds": duration, "stdout_path": self._display(artifact_dir / "stdout.log"),
            "stderr_path": self._display(artifact_dir / "stderr.log"),
            "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-2000:],
            "source_metrics_path": self._display(metrics_path), "verification_errors": errors,
        }
        write_json(artifact_dir / "execution.json", execution)
        metric_name = self.config["objective"]["metric"]
        metric_value = metrics.get(metric_name) if isinstance(metrics, dict) else "n/a"
        outcome = "failed" if errors else "finished"
        self.reporter.emit(
            "EXP",
            f"{label} {outcome} in {duration:.1f}s, returncode={returncode}, {metric_name}={metric_value}",
        )
        if self.reporter.mode == "verbose" and (stdout[-500:].strip() or stderr[-500:].strip()):
            tail = stderr[-500:].strip() or stdout[-500:].strip()
            self.reporter.emit("OUTPUT", " ".join(tail.splitlines())[:500], verbose=True)

        if kind == "baseline":
            self.state["baseline"] = {
                "status": "failed" if errors else "success", "config": candidate_config,
                "metrics": metrics, "execution": execution,
            }
            if not errors:
                self.state["current_config"] = copy.deepcopy(candidate_config)
                self.state["best_config"] = copy.deepcopy(candidate_config)
                self.state["best_metrics"] = metrics
                self.state["best_source"] = "baseline"
                self.state["tested_fingerprints"] = [fingerprint(candidate_config)]
                self.state["phase"] = "experiment"
        else:
            pending = {
                "iteration": self.state["iteration"], "hypothesis": _required_text(args, "hypothesis"),
                "expected_effect": _required_text(args, "expected_effect"),
                "config_candidate": candidate_config, "changes": changes,
                "metrics": metrics, "execution": execution, "verification_errors": errors,
                "snapshot_id": snapshot_id, "evaluation": None,
            }
            self.state["pending_run"] = pending
        if errors:
            self.state["failures"].append({
                "tool_call": self.state["tool_call_count"], "iteration": self.state["iteration"],
                "type": "execution_failure", "message": "; ".join(errors),
                "recovery_action": "LLM must evaluate and restore the verified snapshot",
            })
        return {
            "experiment_kind": kind, "iteration": self.state["iteration"] if kind == "candidate" else 0,
            "returncode": returncode, "timed_out": timed_out, "metrics": metrics,
            "verification_errors": errors, "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-2000:],
            "snapshot_id": snapshot_id,
            "next_action": "inspect metrics/logs, then call evaluate_result" if kind == "candidate" else "inspect baseline and plan the first candidate",
        }

    def _tool_evaluate_result(self, args, _call_dir):
        """Reconcile the LLM decision with independent metric validation."""
        pending = self.state.get("pending_run")
        if not pending or pending["iteration"] != int(args["iteration"]):
            raise ToolError("No matching candidate experiment is awaiting evaluation")
        if pending.get("evaluation"):
            raise ToolError("The pending experiment has already been evaluated")
        # Tool output is evidence, not truth: the harness derives its own
        # decision from process status and persisted metrics.
        if pending["verification_errors"]:
            harness_decision = "rollback"
            harness_reason = "; ".join(pending["verification_errors"])
        else:
            try:
                harness_decision, harness_reason = compare_metrics(
                    pending["metrics"], self.state["best_metrics"], self.config["objective"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                harness_decision = "rollback"
                harness_reason = f"metric comparison failed: {exc}"
            else:
                if harness_decision == "reject":
                    harness_decision = "rollback"
        llm_decision = args["decision"]
        if llm_decision == "accept" and harness_decision == "rollback":
            raise ToolError(f"Independent validation requires rollback: {harness_reason}")
        evaluation = {
            "llm_decision": llm_decision, "llm_reason": _required_text(args, "reason"),
            "evidence": _required_text(args, "evidence"),
            "harness_decision": harness_decision, "harness_reason": harness_reason,
        }
        pending["evaluation"] = evaluation
        self.reporter.emit(
            "CHECK", f"Iteration {pending['iteration']} -> {llm_decision}: {harness_reason}"
        )
        if llm_decision == "accept":
            self.state["current_config"] = copy.deepcopy(pending["config_candidate"])
            self.state["best_config"] = copy.deepcopy(pending["config_candidate"])
            self.state["best_metrics"] = pending["metrics"]
            self.state["best_source"] = f"iteration_{pending['iteration']:02d}"
            self._finalize_pending("accept", None)
            return {"decision": "accept", "harness_reason": harness_reason, "next_action": "plan the next experiment"}
        return {
            "decision": "rollback", "harness_reason": harness_reason,
            "snapshot_id": pending["snapshot_id"], "next_action": "call restore_snapshot before any new experiment",
        }

    def _tool_restore_snapshot(self, args, _call_dir):
        """Restore and read-back verify the accepted pre-candidate snapshot."""
        pending = self.state.get("pending_run")
        if not pending or not pending.get("evaluation"):
            raise ToolError("No evaluated candidate is awaiting restoration")
        if pending["evaluation"]["llm_decision"] != "rollback":
            raise ToolError("The pending evaluation did not request rollback")
        if args.get("snapshot_id") != pending["snapshot_id"]:
            raise ToolError("snapshot_id does not match the pending experiment")
        snapshot = self.run_dir / "snapshots" / pending["snapshot_id"]
        expected = load_yaml(snapshot)
        if expected != self.state["current_config"]:
            raise ToolError("Snapshot differs from canonical accepted_config")
        shutil.copy2(snapshot, self.current_path)
        restored = load_yaml(self.current_path)
        if restored != expected:
            raise ToolError("Restored configuration failed read-back verification")
        qualifies = pending["evaluation"]["harness_decision"] == "rollback"
        recovery = {
            "type": "experiment_rollback", "iteration": pending["iteration"],
            "reason": _required_text(args, "reason"), "snapshot_id": pending["snapshot_id"],
            "sha256": _sha256(self.current_path), "qualifies": qualifies,
        }
        self.state["recovery_events"].append(recovery)
        self._finalize_pending("rollback", recovery)
        self.reporter.emit(
            "RECOVER", f"Iteration {pending['iteration']} restored and hash-verified ({pending['snapshot_id']})"
        )
        return {"restored": True, "verified": True, "recovery": recovery, "next_action": "reflect and plan a different candidate"}

    def _tool_finish(self, args, _call_dir):
        """Complete the run only after plan, experiment, recovery, and report checks."""
        if self.state.get("pending_run"):
            raise ToolError("Resolve the pending experiment before finishing")
        incomplete_steps = [
            step["id"] for step in self.state["plan"]["steps"]
            if step.get("status") != "completed"
        ]
        if incomplete_steps:
            raise ToolError("Complete all plan steps before finish: " + ", ".join(incomplete_steps))
        minimum = self.state["limits"]["min_iterations"]
        if self.state["iteration"] < minimum:
            raise ToolError(f"At least {minimum} candidate experiments are required before finish")
        if self.config.get("evidence", {}).get("require_recovery_event", True):
            if not any(
                event.get("qualifies") and event.get("type") == "experiment_rollback"
                for event in self.state["recovery_events"]
            ):
                raise ToolError("A measured candidate failure and verified experiment rollback are required before finish")
        report_path = self._resolve_write(args["report_path"])
        errors = validate_report(report_path, self.state)
        if errors:
            raise ToolError("Report validation failed: " + "; ".join(errors))
        self.state["requirements"] = {
            "min_iterations_met": True, "recovery_demonstrated": True, "report_verified": True,
        }
        self.state["status"] = "completed"
        self.state["phase"] = "finished"
        self.state["stop_reason"] = "agent_finish"
        self.state["agent_summary"] = _required_text(args, "summary")
        self.reporter.emit("DONE", f"report verified: {self._display(report_path)}")
        return {"finished": True, "status": "completed", "report_path": self._display(report_path)}

    def _finalize_pending(self, decision, recovery):
        """Move a resolved candidate from pending state into immutable history."""
        pending = self.state["pending_run"]
        item = {
            "iteration": pending["iteration"], "hypothesis": pending["hypothesis"],
            "expected_effect": pending["expected_effect"], "changes": pending["changes"],
            "config_candidate": pending["config_candidate"], "metrics": pending["metrics"],
            "execution": pending["execution"], "verification_errors": pending["verification_errors"],
            "decision": decision, "evaluation": pending["evaluation"], "recovery": recovery,
        }
        self.state["history"].append(item)
        self.state["pending_run"] = None

    def _validate_command(self, args, kind):
        """Restrict subprocess execution to the configured experiment protocol."""
        argv = args.get("argv")
        if not isinstance(argv, list) or len(argv) < 2 or not all(isinstance(x, str) for x in argv):
            raise ToolError("argv must be a non-empty string array")
        if any(any(marker in token for marker in ("|", ";", "&&", "||", ">", "<")) for token in argv):
            raise ToolError("Shell control operators are not allowed")
        if argv[0].lower() not in {"python", "python.exe", sys.executable.lower()}:
            raise ToolError("Only the configured Python executable is allowed")
        configured_script = str(self.config["experiment"]["command"][1])
        requested_script = (self.root / argv[1]).resolve() if not Path(argv[1]).is_absolute() else Path(argv[1]).resolve()
        expected_script = (self.root / configured_script).resolve()
        if requested_script != expected_script:
            raise ToolError(f"Only the configured experiment script is allowed: {configured_script}")
        config_arg = _argument_after(argv, "--config")
        requested_config = (self.root / config_arg).resolve() if not Path(config_arg).is_absolute() else Path(config_arg).resolve()
        if requested_config != self.current_path:
            raise ToolError("--config must point to this run's current_config.yaml")
        requested_id = _argument_after(argv, "--run-id")
        if kind == "baseline":
            expected_id = f"{self.state['run_id']}/baseline/raw"
        else:
            expected_id = f"{self.state['run_id']}/iteration_{self.state['iteration'] + 1:02d}/raw"
        if requested_id != expected_id:
            raise ToolError(f"--run-id must be {expected_id}")
        if "--timeout-seconds" in argv:
            command_timeout = float(_argument_after(argv, "--timeout-seconds"))
            if command_timeout > float(args["timeout_seconds"]):
                raise ToolError("Command timeout cannot exceed tool timeout_seconds")
        return [sys.executable, str(requested_script), *argv[2:]], requested_id

    def _bounded_timeout(self, requested):
        """Clamp a tool timeout to experiment and remaining wall-time budgets."""
        maximum = self.state["limits"]["experiment_timeout_seconds"]
        elapsed = time.time() - self.state["started_unix"]
        wall_remaining = self.state["limits"]["max_wall_time_minutes"] * 60 - elapsed
        if requested > maximum:
            raise ToolError(f"timeout_seconds exceeds configured maximum {maximum}")
        if wall_remaining <= 0:
            raise ToolError("Wall-time budget is exhausted")
        return min(requested, maximum, max(0.1, wall_remaining))

    def _read_roots(self):
        """Resolve the configured read allowlist for this run."""
        configured = self.config.get("tools", {}).get(
            "read_roots", ["tasks", "configs", f"runs/{self.state['run_id']}"]
        )
        roots = []
        for value in configured:
            value = str(value).format(run_id=self.state["run_id"])
            path = Path(value)
            roots.append((self.root / path).resolve() if not path.is_absolute() else path.resolve())
        if self.run_dir not in roots:
            roots.append(self.run_dir)
        return roots

    def _resolve_read(self, value):
        """Resolve a requested read path and enforce the read allowlist."""
        path = Path(str(value))
        resolved = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not self._is_readable(resolved):
            raise ToolError(f"Read path is outside allowed roots: {value}")
        return resolved

    def _is_readable(self, path):
        """Return whether a path is contained by any approved read root."""
        resolved = Path(path).resolve()
        return any(_within(resolved, root) for root in self.read_roots)

    def _resolve_write(self, value):
        """Resolve a write target and enforce run-directory containment."""
        path = Path(str(value))
        resolved = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not _within(resolved, self.write_root):
            raise ToolError(f"Write path is outside this run directory: {value}")
        return resolved

    def _display(self, path):
        """Format a path relative to the workspace when possible."""
        try:
            return Path(path).resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(Path(path).resolve())


class ToolError(RuntimeError):
    """Represent a recoverable, LLM-visible tool validation error."""


def _tool(name, description, properties, required):
    """Build an OpenAI function-tool schema with strict object arguments."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
    }}


def _string():
    return {"type": "string", "minLength": 1}


def _required_text(mapping, key):
    """Return a stripped required string or raise a tool validation error."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{key} must be a non-empty string")
    return value.strip()


def _within(path, root):
    """Return whether a resolved path is contained by a resolved root."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _sha256(path):
    """Hash a file for write and rollback verification records."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _argument_after(argv, flag):
    """Extract a required value from a validated argument vector."""
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise ToolError(f"Required command argument is missing: {flag}") from exc


def _config_changes(before, after, prefix=""):
    """Return leaf-level configuration changes keyed by dotted paths."""
    changes = {}
    keys = set(before) | set(after)
    for key in keys:
        dotted = f"{prefix}.{key}" if prefix else key
        if key not in before or key not in after:
            changes[dotted] = after.get(key)
        elif isinstance(before[key], dict) and isinstance(after[key], dict):
            changes.update(_config_changes(before[key], after[key], dotted))
        elif before[key] != after[key]:
            changes[dotted] = after[key]
    return changes


def _validate_changes(changes, search_space):
    """Validate candidate changes against the configured search space."""
    errors = []
    for key, value in changes.items():
        if key not in search_space:
            errors.append(f"{key} is not modifiable")
        elif value not in search_space[key]:
            errors.append(f"{key}={value!r} is not an allowed value")
    return errors


def _yaml_text(data):
    """Serialize configuration data with stable key ordering."""
    import yaml
    return yaml.safe_dump(data, sort_keys=False)


def _decode_timeout(value):
    """Normalize timeout-captured subprocess output to text."""
    value = value or ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
