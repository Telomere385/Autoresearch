import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import yaml


TASK_DIR = Path(__file__).resolve().parent
ROOT_DIR = TASK_DIR.parents[1]
RUNS_DIR = ROOT_DIR / "runs"
SRC_DIR = TASK_DIR / "src"

REQUIRED_KEYS = (
    "seed",
    "training_episodes",
    "evaluation_episodes",
    "learning_rate",
    "discount_factor",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay",
    "state_discretization",
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return data


def write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_value(value):
    try:
        return yaml.safe_load(value)
    except Exception:
        return value


def apply_overrides(config, overrides):
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        config[key] = parse_value(value)
    return config


def require_config(config):
    missing = [key for key in REQUIRED_KEYS if key not in config]
    state = config.get("state_discretization", {})
    if isinstance(state, dict):
        if "dx_dy_bin_size" not in state:
            missing.append("state_discretization.dx_dy_bin_size")
        if "velocity_bin_size" not in state:
            missing.append("state_discretization.velocity_bin_size")
    if missing:
        raise ValueError("Missing required config keys: " + ", ".join(missing))


def normalize_config(config):
    require_config(config)
    rewards = config.get("rewards", {})
    environment = config.get("environment", {})
    state = config["state_discretization"]
    normalized = dict(config)
    normalized["gamma"] = config["discount_factor"]
    normalized["r"] = state["dx_dy_bin_size"]
    normalized["rv"] = state["velocity_bin_size"]
    if "pipe_passed" in rewards:
        normalized["pipe_passed_reward"] = rewards["pipe_passed"]
    if "alive" in rewards:
        normalized["did_not_die_reward"] = rewards["alive"]
    if "death" in rewards:
        normalized["die_reward"] = rewards["death"]
    if "player_flap_acc" in environment:
        normalized["player_flap_acc"] = environment["player_flap_acc"]
    return normalized


def extract_hyperparameters(config):
    keys = (
        "learning_rate",
        "learning_rate_min",
        "learning_rate_decay",
        "discount_factor",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay",
        "sample_t",
        "state_discretization",
        "rewards",
        "environment",
    )
    return {key: config[key] for key in keys if key in config}


def result_from_error(mode, config, started_at, exc):
    return {
        "status": "error",
        "mode": mode,
        "mean_score": 0.0,
        "std_score": 0.0,
        "max_score": 0,
        "evaluation_episodes": int(config.get("evaluation_episodes", 0)) if isinstance(config, dict) else 0,
        "seed": config.get("seed") if isinstance(config, dict) else None,
        "training_time": time.time() - started_at if mode == "train" else 0.0,
        "hyperparameters": extract_hyperparameters(config) if isinstance(config, dict) else {},
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def single_result_path(run_id):
    return RUNS_DIR / run_id / "result.json"


def run_single(mode, config_path, run_id, overrides=None):
    overrides = overrides or []
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    result_path = run_dir / "result.json"
    started_at = time.time()
    config = {}

    with open(log_path, "w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)), contextlib.redirect_stderr(Tee(sys.stderr, log_file)):
            try:
                os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
                os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
                os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

                config = apply_overrides(load_yaml(config_path), overrides)
                config = normalize_config(config)
                if mode == "eval":
                    config["q_init"] = "none"

                shutil.copy2(config_path, run_dir / "source_config.yaml")
                write_json(run_dir / "config.json", config)

                old_argv = sys.argv[:]
                sys.path.insert(0, str(SRC_DIR))
                sys.argv = ["flappyq.py"]
                try:
                    import flappyq
                    result = flappyq.run_configured_experiment(config, mode, run_dir)
                finally:
                    sys.argv = old_argv
                    if str(SRC_DIR) in sys.path:
                        sys.path.remove(str(SRC_DIR))

                result["status"] = "success"
                result["seed"] = config["seed"]
                result["hyperparameters"] = extract_hyperparameters(config)
                result["result_path"] = str(result_path)
                write_json(result_path, result)
                write_json(run_dir / "metrics.json", result)
            except Exception as exc:
                result = result_from_error(mode, config, started_at, exc)
                write_json(result_path, result)
                print(traceback.format_exc())

            print(json.dumps({
                "run_dir": str(run_dir),
                "result_path": str(result_path),
                "status": result["status"],
                "mean_score": result["mean_score"],
                "max_score": result["max_score"],
            }, indent=2))
    return result


def is_success(result):
    return result.get("status") in {"ok", "success"}


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


def call_self(config_path, mode, run_id, log_file, timeout_seconds):
    cmd = [
        sys.executable,
        str(TASK_DIR / "run.py"),
        "--mode",
        mode,
        "--config",
        str(config_path),
        "--run-id",
        run_id,
    ]
    started = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(0.1, timeout_seconds),
        )
        returncode = completed.returncode
        output = completed.stdout
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        output += f"\n{mode} timed out after {timeout_seconds:.1f} seconds.\n"
    elapsed = time.time() - started
    log_file.write("\n$ " + " ".join(cmd) + "\n")
    log_file.write(output)
    log_file.flush()
    return returncode, elapsed


def read_result(path):
    if not path.exists():
        return {
            "status": "error",
            "error": {
                "type": "MissingResult",
                "message": f"Missing result file: {path}",
            },
        }
    return json.loads(path.read_text(encoding="utf-8"))


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
        "mean_reward": float(eval_result.get("mean_reward", 0.0)) if eval_ok else 0.0,
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


def run_train_eval(config_path, run_id, timeout_seconds):
    started_at = time.time()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_dir / "summary.json"
    try:
        config = load_yaml(config_path)
        config["seed"] = int(config.get("seed", 0))
        write_yaml(run_dir / "config.yaml", config)
        train_config_path, eval_config_path = build_phase_configs(config, run_dir)
        deadline = time.time() + timeout_seconds
        with open(run_dir / "experiment.log", "w", encoding="utf-8") as log_file:
            train_code, train_wall_time = call_self(
                train_config_path, "train", f"{run_id}/train", log_file, deadline - time.time()
            )
            train_result = read_result(single_result_path(f"{run_id}/train"))
            eval_code = None
            eval_wall_time = 0.0
            if train_code == 0 and is_success(train_result):
                eval_code, eval_wall_time = call_self(
                    eval_config_path, "eval", f"{run_id}/eval", log_file, deadline - time.time()
                )
                eval_result = read_result(single_result_path(f"{run_id}/eval"))
            else:
                eval_result = {
                    "status": "error",
                    "error": {
                        "type": "SkippedEvaluation",
                        "message": "Training failed, so evaluation was skipped.",
                    },
                }
        summary = summarize(run_id, started_at, config, train_code, train_wall_time, train_result, eval_code, eval_wall_time, eval_result)
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
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the Flappy Bird Q-learning task adapter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mode", choices=["train", "eval"], default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()

    run_id = args.run_id or f"flappy_qlearning_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.mode:
        result = run_single(args.mode, Path(args.config), run_id, args.overrides)
    else:
        result = run_train_eval(Path(args.config), run_id, args.timeout_seconds)
    if not is_success(result):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
