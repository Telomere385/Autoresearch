# 这是工程内置的 Git 辅助工具。
# 它为 Agent 和命令行提供受控的 status、commit、文件 restore 和 reset-to-commit 操作封装。
import argparse
import json
import subprocess
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


class GitError(RuntimeError):
    pass


def run_git(args, cwd=ROOT_DIR, check=True):
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result = {
        "command": ["git", *args],
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if check and completed.returncode != 0:
        raise GitError(result["stderr"] or result["stdout"] or "git command failed")
    return result


def repo_status():
    return run_git(["status", "--short"], check=True)


def commit_paths(paths, message, push=False, remote="origin", branch=None):
    normalized = [normalize_repo_path(path) for path in paths if path]
    if not normalized:
        return {"status": "skipped", "reason": "no paths to commit"}

    actions = []
    actions.append(run_git(["add", "--", *normalized]))
    diff = run_git(["diff", "--cached", "--quiet"], check=False)
    if diff["returncode"] == 0:
        return {"status": "skipped", "reason": "no staged changes", "paths": normalized, "actions": actions}

    actions.append(run_git(["commit", "-m", message]))
    commit_hash = run_git(["rev-parse", "--short", "HEAD"])["stdout"]
    result = {"status": "committed", "commit": commit_hash, "paths": normalized, "actions": actions}
    if push:
        push_args = ["push", remote]
        if branch:
            push_args.append(branch)
        actions.append(run_git(push_args))
        result["status"] = "pushed"
        result["remote"] = remote
        result["branch"] = branch
    return result


def rollback_paths(paths):
    normalized = [normalize_repo_path(path) for path in paths if path]
    if not normalized:
        return {"status": "skipped", "reason": "no paths to rollback"}
    actions = [
        run_git(["restore", "--staged", "--worktree", "--", *normalized]),
    ]
    return {"status": "rolled_back", "paths": normalized, "actions": actions}


def rollback_to_commit(commit):
    if not commit:
        raise GitError("rollback_to_commit requires a commit hash")
    actions = [run_git(["reset", "--hard", commit])]
    return {"status": "rolled_back", "commit": commit, "actions": actions}


def normalize_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        path = path.resolve().relative_to(ROOT_DIR)
    return path.as_posix()


def main():
    parser = argparse.ArgumentParser(description="Small Git helper for AutoResearch task versioning.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")

    commit_parser = sub.add_parser("commit")
    commit_parser.add_argument("--message", required=True)
    commit_parser.add_argument("--push", action="store_true")
    commit_parser.add_argument("--remote", default="origin")
    commit_parser.add_argument("--branch", default=None)
    commit_parser.add_argument("paths", nargs="+")

    rollback_parser = sub.add_parser("rollback")
    rollback_parser.add_argument("paths", nargs="+")

    reset_parser = sub.add_parser("reset-to")
    reset_parser.add_argument("--commit", required=True)

    args = parser.parse_args()

    if args.command == "status":
        result = repo_status()
    elif args.command == "commit":
        result = commit_paths(args.paths, args.message, push=args.push, remote=args.remote, branch=args.branch)
    elif args.command == "rollback":
        result = rollback_paths(args.paths)
    else:
        result = rollback_to_commit(args.commit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
