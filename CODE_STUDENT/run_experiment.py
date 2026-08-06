import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
RUNS_DIR = BASE_DIR / "runs"


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


def load_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("YAML config must contain a mapping at the top level.")
    return config


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_value(value):
    import yaml

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


def build_run_dir(mode, run_id):
    if run_id is None:
        run_id = f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


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


def result_from_error(mode, config, started_at, exc):
    seed = config.get("seed") if isinstance(config, dict) else None
    return {
        "status": "error",
        "mode": mode,
        "mean_score": 0.0,
        "std_score": 0.0,
        "max_score": 0,
        "evaluation_episodes": int(config.get("evaluation_episodes", 0)) if isinstance(config, dict) else 0,
        "seed": seed,
        "training_time": time.time() - started_at if mode == "train" else 0.0,
        "hyperparameters": extract_hyperparameters(config),
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def extract_hyperparameters(config):
    if not isinstance(config, dict):
        return {}
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


def main():
    parser = argparse.ArgumentParser(description="Run a reproducible Flappy Bird Q-learning experiment.")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--config", default=str(BASE_DIR / "configs" / "default.yaml"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="Override YAML config values, e.g. --set epsilon_start=0.01")
    args = parser.parse_args()

    run_dir = build_run_dir(args.mode, args.run_id)
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

                config_path = Path(args.config)
                config = apply_overrides(load_config(config_path), args.overrides)
                config = normalize_config(config)
                if args.mode == "eval":
                    config["q_init"] = "none"

                shutil.copy2(config_path, run_dir / "source_config.yaml")
                write_json(run_dir / "config.json", config)

                old_argv = sys.argv[:]
                sys.path.insert(0, str(SRC_DIR))
                sys.argv = ["flappyq.py"]
                try:
                    import flappyq
                finally:
                    sys.argv = old_argv

                result = flappyq.run_configured_experiment(config, args.mode, run_dir)
                result["status"] = "ok"
                result["seed"] = config["seed"]
                result["hyperparameters"] = extract_hyperparameters(config)
                result["result_path"] = str(result_path)
                write_json(result_path, result)
                write_json(run_dir / "metrics.json", result)

            except Exception as exc:
                result = result_from_error(args.mode, config, started_at, exc)
                write_json(result_path, result)
                print(traceback.format_exc())

            print(json.dumps({
                "run_dir": str(run_dir),
                "result_path": str(result_path),
                "status": result["status"],
                "mean_score": result["mean_score"],
                "max_score": result["max_score"],
            }, indent=2))


if __name__ == "__main__":
    main()
