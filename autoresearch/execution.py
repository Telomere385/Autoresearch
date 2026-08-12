import json
import math
from pathlib import Path

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


def fingerprint(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


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


def _require_numeric_metric(metrics, name):
    if name not in metrics:
        raise ValueError(f"Tie-breaker metric is missing: {name}")
    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError(f"Tie-breaker metric is not finite: {name}")
