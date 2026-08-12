import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml


def load_yaml(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain an object: {path}")
    return data


def write_yaml(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def apply_changes(config, changes):
    updated = copy.deepcopy(config)
    for dotted_key, value in changes.items():
        target = updated
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Cannot set {dotted_key}: {part} is not an object")
            target = target[part]
        target[parts[-1]] = value
    return updated


def validate_proposal(proposal, current_config, search_space, tested_fingerprints):
    errors = []
    changes = proposal.get("changes")
    if not isinstance(changes, dict) or not changes:
        return ["changes must be a non-empty object"], None
    for key, value in changes.items():
        if key not in search_space:
            errors.append(f"{key} is not modifiable")
        elif value not in search_space[key]:
            errors.append(f"{key}={value!r} is not an allowed value")
    if errors:
        return errors, None
    try:
        candidate = apply_changes(current_config, changes)
    except ValueError as exc:
        return [str(exc)], None
    if fingerprint(candidate) in tested_fingerprints:
        return ["candidate repeats a previously tested configuration"], None
    return [], candidate


def fingerprint(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def activate_config(source, current_path):
    """Copy and read back a config so file modification or rollback is verified."""
    shutil.copy2(source, current_path)
    expected = load_yaml(source)
    actual = load_yaml(current_path)
    if actual != expected:
        raise RuntimeError(f"Config verification failed after copying {source} to {current_path}")
    return actual


def run_experiment(root, runtime_config, config_path, experiment_id, artifact_dir, timeout_seconds=None):
    experiment = runtime_config["experiment"]
    timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else runtime_config["budget"]["experiment_timeout_seconds"]
    )
    values = {
        "python": sys.executable,
        "config_path": str(Path(config_path).resolve()),
        "experiment_id": experiment_id,
        "timeout_seconds": str(timeout),
    }
    command = [str(part).format(**values) for part in experiment["command"]]
    started = time.time()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout + 5,
        )
        returncode = completed.returncode
        output = completed.stdout
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        output = _timeout_output(exc)

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "stdout.log").write_text(output, encoding="utf-8")
    metrics_path = root / experiment["metrics_path"].format(experiment_id=experiment_id)
    metrics, errors = verify_execution(returncode, timed_out, metrics_path, runtime_config["objective"])
    if metrics is not None:
        write_json(artifact_dir / "metrics.json", metrics)
    result = {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_seconds": time.time() - started,
        "stdout_path": str(artifact_dir / "stdout.log"),
        "stdout_tail": output[-2000:],
        "source_metrics_path": str(metrics_path),
        "metrics_path": str(artifact_dir / "metrics.json") if metrics is not None else None,
        "verification_errors": errors,
    }
    write_json(artifact_dir / "execution.json", result)
    return result, metrics


def verify_execution(returncode, timed_out, metrics_path, objective):
    errors = []
    if timed_out:
        errors.append("experiment timed out")
    if returncode != 0:
        errors.append(f"experiment exited with return code {returncode}")
    metrics = None
    if not Path(metrics_path).exists():
        errors.append(f"metrics file is missing: {metrics_path}")
        return metrics, errors
    try:
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"metrics file is unreadable: {exc}")
        return None, errors
    if not isinstance(metrics, dict):
        errors.append("metrics must be a JSON object")
        return metrics, errors
    if metrics.get("status") != "success":
        errors.append(f"metrics status is {metrics.get('status')!r}, not 'success'")
    metric_name = objective["metric"]
    if metric_name not in metrics:
        errors.append(f"required metric is missing: {metric_name}")
    else:
        try:
            value = float(metrics[metric_name])
            if not math.isfinite(value):
                errors.append(f"metric {metric_name} is not finite")
        except (TypeError, ValueError):
            errors.append(f"metric {metric_name} is not numeric")
    return metrics, errors


def compare_metrics(candidate, best, objective):
    metric = objective["metric"]
    direction = objective.get("direction", "maximize")
    minimum = float(objective.get("min_improvement", 0.0))
    candidate_value = float(candidate[metric])
    best_value = float(best[metric])
    delta = candidate_value - best_value if direction == "maximize" else best_value - candidate_value
    if delta > minimum:
        return "accept", f"{metric} improved from {best_value} to {candidate_value}"

    tie_metric = objective.get("tie_breaker_metric")
    if tie_metric and abs(candidate_value - best_value) <= minimum:
        _require_numeric_metric(candidate, tie_metric)
        _require_numeric_metric(best, tie_metric)
        tie_delta = float(candidate[tie_metric]) - float(best[tie_metric])
        if direction == "minimize":
            tie_delta = -tie_delta
        if tie_delta > float(objective.get("tie_breaker_min_improvement", 0.0)):
            return "accept", (
                f"{metric} tied at {candidate_value}; {tie_metric} improved "
                f"from {best[tie_metric]} to {candidate[tie_metric]}"
            )
    return "reject", f"{metric} did not improve over {best_value}"


def target_reached(metrics, objective):
    target = objective.get("target")
    if target is None or metrics is None:
        return False
    value = float(metrics[objective["metric"]])
    return value >= float(target) if objective.get("direction", "maximize") == "maximize" else value <= float(target)


def _require_numeric_metric(metrics, name):
    if name not in metrics:
        raise ValueError(f"Tie-breaker metric is missing: {name}")
    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError(f"Tie-breaker metric is not finite: {name}")


def _timeout_output(exc):
    output = exc.stdout or ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    return output + f"\nExperiment timed out after {exc.timeout} seconds.\n"
