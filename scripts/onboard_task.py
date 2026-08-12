# 这是任务接入脚本。
# 它扫描 tasks/<task>/project 中的外部项目，调用 LLM 生成 task manifest、agent config、默认实验配置、目标文件和 runner。
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from planner.base import PlannerError
from planner.providers import call_provider
from scripts.vcs import commit_paths


TASKS_DIR = ROOT_DIR / "tasks"

GENERATED_PATHS = {
    "README.md",
    "requirements.txt",
    "manifest/task.yaml",
    "configs/agent.yaml",
    "configs/experiment.yaml",
    "configs/goal.md",
    "runner/run.py",
}

TEXT_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".txt",
    ".md",
    ".csv",
    ".sh",
    ".ps1",
    ".bat",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    "runs",
    "reports",
}


def main():
    parser = argparse.ArgumentParser(description="Use an LLM to convert an imported project into an AutoResearch task.")
    parser.add_argument("--task", required=True, help="Task folder name under tasks/.")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "local", "local_endpoint", "local_command"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--local-endpoint", default=None)
    parser.add_argument("--local-command", default=None)
    parser.add_argument("--max-files", type=int, default=40)
    parser.add_argument("--max-bytes-per-file", type=int, default=8000)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--force", action="store_true", help="Overwrite generated adapter files after backing them up.")
    parser.add_argument("--auto-commit", action="store_true", help="Commit generated adapter files after validation.")
    parser.add_argument("--push", action="store_true", help="Push the generated commit to GitHub remote.")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default=None)
    parser.add_argument("--commit-message", default=None)
    args = parser.parse_args()

    started = time.time()
    task_dir = TASKS_DIR / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_dir / "onboarding_log.json"

    try:
        snapshot = scan_task_dir(task_dir, args.max_files, args.max_bytes_per_file)
        planner_config = build_planner_config(args)
        raw = call_provider(planner_config, build_prompt(args.task, snapshot))
        plan = parse_response(raw)
        validate_plan(plan)
        written = write_generated_files(task_dir, plan["files"], force=args.force)
        validate_generated_task(task_dir)
        vcs = maybe_commit(args, task_dir, written)
        status = "success"
        error = None
    except Exception as exc:
        status = "error"
        error = {"type": type(exc).__name__, "message": str(exc)}
        written = []
        vcs = {"status": "skipped", "reason": "onboarding failed"}
        raw = locals().get("raw")
        plan = locals().get("plan")
        snapshot = locals().get("snapshot", {})

    log = {
        "status": status,
        "task": args.task,
        "duration_seconds": time.time() - started,
        "provider": args.provider,
        "model": args.model,
        "snapshot": summarize_snapshot(snapshot),
        "raw_response": raw,
        "parsed_plan": plan,
        "written_files": written,
        "vcs": vcs,
        "error": error,
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "task_dir": str(task_dir),
        "log": str(log_path),
        "written_files": written,
        "vcs": vcs,
        "error": error,
    }, indent=2))
    if status != "success":
        raise SystemExit(1)


def build_planner_config(args):
    config = {
        "provider": args.provider,
        "model": args.model,
        "timeout_seconds": args.timeout_seconds,
        "temperature": args.temperature,
    }
    if args.api_key_env:
        config["api_key_env"] = args.api_key_env
    if args.base_url:
        config["base_url"] = args.base_url
    if args.local_endpoint:
        config["local_endpoint"] = args.local_endpoint
    if args.local_command:
        config["local_command"] = args.local_command
    return config


def scan_task_dir(task_dir, max_files, max_bytes_per_file):
    files = []
    for path in sorted(task_dir.rglob("*")):
        rel = path.relative_to(task_dir).as_posix()
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(task_dir).parts):
            continue
        if rel in GENERATED_PATHS or rel == "onboarding_log.json":
            continue
        stat = path.stat()
        entry = {"path": rel, "size": stat.st_size}
        if path.suffix.lower() in TEXT_SUFFIXES and len(files) < max_files:
            try:
                entry["content"] = path.read_text(encoding="utf-8")[:max_bytes_per_file]
            except UnicodeDecodeError:
                entry["content"] = path.read_text(encoding="latin-1")[:max_bytes_per_file]
        files.append(entry)
    return {"files": files[:max_files], "total_seen": len(files)}


def build_prompt(task_name, snapshot):
    payload = {
        "role": "You are integrating an existing small ML/RL project into a generic AutoResearch Agent.",
        "task_name": task_name,
        "repository_contract": {
            "generated_files": sorted(GENERATED_PATHS),
            "runner_contract": [
                "runner/run.py must accept --config and --run-id.",
                "runner/run.py must execute a real train/eval command or call existing code in the imported project.",
                "runner/run.py must write runs/<task>/<run_id>/summary.json.",
                "summary.json must include status, mean_score, max_score, mean_reward, total_training_time.",
                "Evaluation must be deterministic when the imported project exposes seed/config controls.",
            ],
            "agent_contract": [
                "configs/agent.yaml must point task_file to tasks/<task>/manifest/task.yaml.",
                "configs/agent.yaml must point goal_file to tasks/<task>/configs/goal.md.",
                "manifest/task.yaml interface.experiment_runner must point to tasks/<task>/runner/run.py.",
                "Only expose safe YAML parameters in search_space/editable_surface.",
            ],
        },
        "hard_rules": [
            "Return only one JSON object. No markdown.",
            "Do not ask the user to edit files manually.",
            "Do not modify imported source files.",
            "Only generate files in generated_files.",
            "Use relative paths from repository root in YAML and runner code.",
            "Prefer wrapping the imported project over rewriting it.",
            "If the project cannot be identified, generate a runner that fails clearly with status=error and explains missing requirements.",
        ],
        "required_output_schema": {
            "summary": "string",
            "files": [
                {
                    "path": "one of generated_files",
                    "content": "complete file content",
                }
            ],
        },
        "imported_project_snapshot": snapshot,
    }
    return json.dumps(payload, indent=2)


def parse_response(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise PlannerError("LLM response did not contain a JSON object.")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise PlannerError("LLM response must be a JSON object.")
    return data


def validate_plan(plan):
    files = plan.get("files")
    if not isinstance(files, list) or not files:
        raise PlannerError("LLM response must include a non-empty files list.")
    seen = set()
    for item in files:
        if not isinstance(item, dict):
            raise PlannerError("Each generated file entry must be an object.")
        rel = item.get("path")
        content = item.get("content")
        if rel not in GENERATED_PATHS:
            raise PlannerError(f"Generated path is not allowed: {rel}")
        if rel in seen:
            raise PlannerError(f"Duplicate generated path: {rel}")
        if not isinstance(content, str) or not content.strip():
            raise PlannerError(f"Generated file is empty: {rel}")
        seen.add(rel)
    required = {"manifest/task.yaml", "configs/agent.yaml", "configs/experiment.yaml", "configs/goal.md", "runner/run.py"}
    missing = required - seen
    if missing:
        raise PlannerError(f"Missing required generated files: {sorted(missing)}")


def write_generated_files(task_dir, files, force):
    backup_dir = task_dir / ".onboarding_backups" / time.strftime("%Y%m%d_%H%M%S")
    written = []
    for item in files:
        rel = item["path"]
        path = task_dir / rel
        if path.exists():
            if not force:
                raise PlannerError(f"{rel} already exists. Re-run with --force to overwrite generated adapter files.")
            backup_path = backup_dir / rel
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item["content"].replace("\r\n", "\n"), encoding="utf-8")
        written.append(rel)
    return written


def maybe_commit(args, task_dir, written):
    if not args.auto_commit and not args.push:
        return {"status": "skipped", "reason": "auto commit disabled"}
    paths = [task_dir / rel for rel in written]
    message = args.commit_message or f"Onboard AutoResearch task: {task_dir.name}"
    try:
        return commit_paths(paths, message, push=args.push, remote=args.remote, branch=args.branch)
    except Exception as exc:
        return {"status": "error", "type": type(exc).__name__, "message": str(exc)}


def validate_generated_task(task_dir):
    manifest = yaml.safe_load((task_dir / "manifest" / "task.yaml").read_text(encoding="utf-8"))
    agent = yaml.safe_load((task_dir / "configs" / "agent.yaml").read_text(encoding="utf-8"))
    experiment = yaml.safe_load((task_dir / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(agent, dict) or not isinstance(experiment, dict):
        raise PlannerError("Generated YAML files must contain objects.")
    runner = task_dir / "runner" / "run.py"
    if not runner.exists():
        raise PlannerError("Generated runner/run.py is missing.")
    interface = manifest.get("interface", {})
    expected_runner = f"tasks/{task_dir.name}/runner/run.py"
    if interface.get("experiment_runner") != expected_runner:
        raise PlannerError(f"manifest interface.experiment_runner must be {expected_runner}")


def summarize_snapshot(snapshot):
    return {
        "total_seen": snapshot.get("total_seen"),
        "included_files": [
            {"path": item.get("path"), "size": item.get("size"), "has_content": "content" in item}
            for item in snapshot.get("files", [])
        ],
    }


if __name__ == "__main__":
    main()
