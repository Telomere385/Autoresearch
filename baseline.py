import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import yaml


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Baseline config must contain a mapping.")
    return data


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def make_run_id(prefix, seed, phase):
    return f"{prefix}/seed_{seed}/{phase}"


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


def aggregate(seed_results):
    scores = [
        float(item["eval_result"]["mean_score"])
        for item in seed_results
        if is_success(item.get("eval_result", {}))
    ]
    total_training_time = sum(
        float(item.get("train_result", {}).get("training_time", 0.0))
        for item in seed_results
    )

    if len(scores) == len(seed_results) and scores:
        status = "success"
    elif scores:
        status = "partial_failure"
    else:
        status = "error"

    return {
        "status": status,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "min_score": float(np.min(scores)) if scores else 0.0,
        "max_score": float(np.max(scores)) if scores else 0.0,
        "total_training_time": float(total_training_time),
        "seed_results": seed_results,
    }


def build_seed_configs(base_config, seed, baseline_dir):
    train_config = dict(base_config)
    train_config.pop("seeds", None)
    train_config["seed"] = seed

    eval_config = dict(train_config)
    eval_config["q_init"] = "none"
    eval_config["epsilon_start"] = 0.0
    eval_config["epsilon_end"] = 0.0
    eval_config["epsilon_decay"] = 1.0
    baseline_rel = baseline_dir.relative_to(RUNS_DIR).as_posix()
    eval_config["input_q_table"] = f"runs/{baseline_rel}/seed_{seed}/train/q_table.npy"

    seed_dir = baseline_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    train_config_path = seed_dir / "train_config.yaml"
    eval_config_path = seed_dir / "eval_config.yaml"
    write_yaml(train_config_path, train_config)
    write_yaml(eval_config_path, eval_config)
    return train_config_path, eval_config_path


def error_summary(run_id, started_at, exc):
    return {
        "status": "error",
        "mean_score": 0.0,
        "std_score": 0.0,
        "min_score": 0.0,
        "max_score": 0.0,
        "total_training_time": 0.0,
        "seed_results": [],
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
        "run_id": run_id,
        "duration_seconds": time.time() - started_at,
    }


def main():
    parser = argparse.ArgumentParser(description="Run train+eval baseline experiments across multiple seeds.")
    parser.add_argument("--config", default=str(BASE_DIR / "configs" / "experiment.yaml"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    started_at = time.time()
    run_id = args.run_id or f"baseline_{time.strftime('%Y%m%d_%H%M%S')}"
    baseline_dir = RUNS_DIR / run_id
    baseline_dir.mkdir(parents=True, exist_ok=False)
    summary_path = baseline_dir / "summary.json"

    try:
        base_config = load_yaml(Path(args.config))
        seeds = base_config.get("seeds", [42, 123, 2026])
        if not seeds:
            raise ValueError("Baseline config must include at least one seed.")

        write_yaml(baseline_dir / "config.yaml", base_config)
        seed_results = []
        with open(baseline_dir / "baseline.log", "w", encoding="utf-8") as log_file:
            for seed in seeds:
                train_config_path, eval_config_path = build_seed_configs(base_config, seed, baseline_dir)
                train_run_id = make_run_id(run_id, seed, "train")
                eval_run_id = make_run_id(run_id, seed, "eval")

                train_code, train_wall_time = call_experiment(train_config_path, "train", train_run_id, log_file)
                train_result_path = RUNS_DIR / train_run_id / "result.json"
                train_result = read_result(train_result_path)

                eval_code = None
                eval_wall_time = 0.0
                eval_result_path = RUNS_DIR / eval_run_id / "result.json"
                if train_code == 0 and is_success(train_result):
                    eval_code, eval_wall_time = call_experiment(eval_config_path, "eval", eval_run_id, log_file)
                    eval_result = read_result(eval_result_path)
                else:
                    eval_result = {
                        "status": "error",
                        "error": {
                            "type": "SkippedEvaluation",
                            "message": "Training failed, so evaluation was skipped.",
                        },
                    }

                seed_results.append({
                    "seed": seed,
                    "train_run_dir": str(RUNS_DIR / train_run_id),
                    "eval_run_dir": str(RUNS_DIR / eval_run_id),
                    "train_status": train_result.get("status", "error"),
                    "eval_status": eval_result.get("status", "error"),
                    "train_returncode": train_code,
                    "eval_returncode": eval_code,
                    "train_wall_time": train_wall_time,
                    "eval_wall_time": eval_wall_time,
                    "training_time": float(train_result.get("training_time", 0.0)),
                    "eval_mean_score": float(eval_result.get("mean_score", 0.0)),
                    "eval_std_score": float(eval_result.get("std_score", 0.0)),
                    "eval_max_score": float(eval_result.get("max_score", 0.0)),
                    "train_result": train_result,
                    "eval_result": eval_result,
                })
                write_json(summary_path, aggregate(seed_results))

        summary = aggregate(seed_results)
        summary["run_id"] = run_id
        summary["duration_seconds"] = time.time() - started_at
        write_json(summary_path, summary)
    except Exception as exc:
        summary = error_summary(run_id, started_at, exc)
        write_json(summary_path, summary)

    print(json.dumps({
        "run_dir": str(baseline_dir),
        "summary_path": str(summary_path),
        "status": summary["status"],
        "mean_score": summary["mean_score"],
        "std_score": summary["std_score"],
        "min_score": summary["min_score"],
        "max_score": summary["max_score"],
    }, indent=2))


if __name__ == "__main__":
    main()
