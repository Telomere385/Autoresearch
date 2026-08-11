import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import yaml


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Experiment config must contain a mapping.")
    return data


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def call_experiment(config_path, mode, run_id, log_file):
    cmd = [
        sys.executable,
        str(BASE_DIR / "experiment.py"),
        "--mode",
        mode,
        "--config",
        str(config_path),
        "--run-id",
        run_id,
    ]
    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - started
    log_file.write("\n$ " + " ".join(cmd) + "\n")
    log_file.write(completed.stdout)
    log_file.flush()
    return completed.returncode, elapsed


def read_result(path):
    if not path.exists():
        return {
            "status": "error",
            "error": {
                "type": "MissingResult",
                "message": f"Missing result file: {path}",
            },
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_success(result):
    return result.get("status") in {"ok", "success"}


def mean_reward(result):
    return float(result.get("mean_reward", 0.0)) if is_success(result) else 0.0


def build_phase_configs(base_config, run_dir):
    config = dict(base_config)
    config["seed"] = int(config.get("seed", 0))

    train_config = dict(config)
    eval_config = dict(config)
    eval_config["q_init"] = "none"
    eval_config["epsilon_start"] = 0.0
    eval_config["epsilon_end"] = 0.0
    eval_config["epsilon_decay"] = 1.0
    eval_config["input_q_table"] = f"runs/{run_dir.relative_to(RUNS_DIR).as_posix()}/train/q_table.npy"

    train_config_path = run_dir / "train_config.yaml"
    eval_config_path = run_dir / "eval_config.yaml"
    write_yaml(train_config_path, train_config)
    write_yaml(eval_config_path, eval_config)
    return train_config_path, eval_config_path


def summarize(run_id, started_at, config, train_code, train_wall_time, train_result, eval_code, eval_wall_time, eval_result):
    train_ok = train_code == 0 and is_success(train_result)
    eval_ok = eval_code == 0 and is_success(eval_result)
    status = "success" if train_ok and eval_ok else "error"
    return {
        "status": status,
        "run_id": run_id,
        "seed": int(config.get("seed", 0)),
        "mean_score": float(eval_result.get("mean_score", 0.0)) if eval_ok else 0.0,
        "std_score": float(eval_result.get("std_score", 0.0)) if eval_ok else 0.0,
        "min_score": float(eval_result.get("mean_score", 0.0)) if eval_ok else 0.0,
        "max_score": float(eval_result.get("max_score", 0.0)) if eval_ok else 0.0,
        "mean_reward": mean_reward(eval_result),
        "total_training_time": float(train_result.get("training_time", 0.0)) if is_success(train_result) else 0.0,
        "duration_seconds": time.time() - started_at,
        "train_run_dir": str(RUNS_DIR / run_id / "train"),
        "eval_run_dir": str(RUNS_DIR / run_id / "eval"),
        "train_status": train_result.get("status", "error"),
        "eval_status": eval_result.get("status", "error"),
        "train_returncode": train_code,
        "eval_returncode": eval_code,
        "train_wall_time": train_wall_time,
        "eval_wall_time": eval_wall_time,
        "train_result": train_result,
        "eval_result": eval_result,
    }


def error_summary(run_id, started_at, exc):
    return {
        "status": "error",
        "run_id": run_id,
        "seed": None,
        "mean_score": 0.0,
        "std_score": 0.0,
        "min_score": 0.0,
        "max_score": 0.0,
        "mean_reward": 0.0,
        "total_training_time": 0.0,
        "duration_seconds": time.time() - started_at,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run one train+eval experiment for a single seed.")
    parser.add_argument("--config", default=str(BASE_DIR / "configs" / "experiment.yaml"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    started_at = time.time()
    run_id = args.run_id or f"experiment_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_dir / "summary.json"

    try:
        config = load_yaml(Path(args.config))
        config["seed"] = int(config.get("seed", 0))
        write_yaml(run_dir / "config.yaml", config)
        train_config_path, eval_config_path = build_phase_configs(config, run_dir)

        with open(run_dir / "experiment.log", "w", encoding="utf-8") as log_file:
            train_code, train_wall_time = call_experiment(train_config_path, "train", f"{run_id}/train", log_file)
            train_result = read_result(RUNS_DIR / run_id / "train" / "result.json")

            eval_code = None
            eval_wall_time = 0.0
            if train_code == 0 and is_success(train_result):
                eval_code, eval_wall_time = call_experiment(eval_config_path, "eval", f"{run_id}/eval", log_file)
                eval_result = read_result(RUNS_DIR / run_id / "eval" / "result.json")
            else:
                eval_result = {
                    "status": "error",
                    "error": {
                        "type": "SkippedEvaluation",
                        "message": "Training failed, so evaluation was skipped.",
                    },
                }

        summary = summarize(run_id, started_at, config, train_code, train_wall_time, train_result, eval_code, eval_wall_time, eval_result)
        write_json(summary_path, summary)
    except Exception as exc:
        summary = error_summary(run_id, started_at, exc)
        write_json(summary_path, summary)

    print(json.dumps({
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "status": summary["status"],
        "mean_score": summary["mean_score"],
        "std_score": summary["std_score"],
        "min_score": summary["min_score"],
        "max_score": summary["max_score"],
    }, indent=2))


if __name__ == "__main__":
    main()
