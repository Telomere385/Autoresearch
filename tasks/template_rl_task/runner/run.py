import argparse
import json
from pathlib import Path
import time


ROOT_DIR = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT_DIR / "runs"


def main():
    parser = argparse.ArgumentParser(description="Template RL task runner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "error",
        "run_id": args.run_id,
        "mean_score": 0.0,
        "max_score": 0.0,
        "mean_reward": 0.0,
        "total_training_time": 0.0,
        "duration_seconds": 0.0,
        "error": {
            "type": "TemplateNotImplemented",
            "message": "Replace tasks/template_rl_task/runner/run.py with a real train/eval runner.",
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "run_dir": str(run_dir),
        "summary_path": str(run_dir / "summary.json"),
        "status": summary["status"],
        "mean_score": summary["mean_score"],
    }, indent=2))


if __name__ == "__main__":
    main()
