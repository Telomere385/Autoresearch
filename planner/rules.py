from .base import BasePlanner


class RulesPlanner(BasePlanner):
    def propose(self, context):
        iteration = context["iteration"]
        best = context.get("best")
        last = context.get("last")
        history = context.get("history", [])
        current_config = context.get("current_config", {})
        diagnosis = _diagnose(history, best)
        target_score = context["objective"].get("target_mean_score")
        under_target = target_score is not None and (
            diagnosis["best_score"] is None or diagnosis["best_score"] < float(target_score)
        )

        if diagnosis["all_scores_zero"] and iteration == 1:
            changes = {
                "training_episodes": _next_allowed(context, "training_episodes", current_config.get("training_episodes", 1)),
                "epsilon_start": 0.05,
                "epsilon_end": 0.001,
                "epsilon_decay": 0.995,
            }
            rationale = "Scores are still zero, so first increase the real training budget and exploration."
        elif diagnosis["all_scores_zero"] and iteration == 2:
            changes = {
                "training_episodes": _next_allowed(context, "training_episodes", last.get("config", current_config).get("training_episodes", 1)),
                "learning_rate": 0.4,
                "discount_factor": 0.99,
                "epsilon_start": 0.05,
                "epsilon_end": 0.0,
                "epsilon_decay": 0.99,
            }
            rationale = "No evaluated policy has passed a pipe yet; allocate more training and favor longer-term reward."
        elif diagnosis["all_scores_zero"] and iteration >= 3:
            changes = {
                "training_episodes": _next_allowed(context, "training_episodes", last.get("config", current_config).get("training_episodes", 1)),
                "sample_t": 2,
                "rewards.alive": 0.1,
                "epsilon_start": 0.01,
                "epsilon_end": 0.0,
            }
            rationale = "Score remains zero after budget increases; test a denser action interval and stronger survival shaping."
        elif under_target and _can_increase(context, "training_episodes", current_config.get("training_episodes", 1)):
            changes = {
                "training_episodes": _next_allowed(context, "training_episodes", current_config.get("training_episodes", 1)),
                "epsilon_start": 0.01,
                "epsilon_end": 0.001,
                "epsilon_decay": 0.995,
            }
            rationale = f"Best score is below target {target_score}; increase training budget before trusting smaller hyperparameter changes."
        elif iteration == 1:
            changes = {
                "epsilon_start": 0.01,
                "epsilon_end": 0.001,
                "epsilon_decay": 0.995,
            }
            rationale = "Increase early exploration while preserving a low final exploration rate."
        elif iteration == 2:
            improved = _improved(last, best, context)
            changes = {
                "epsilon_start": 0.01 if improved else 0.001,
                "epsilon_end": 0.001,
                "epsilon_decay": 0.995 if improved else 1.0,
                "learning_rate": 0.7 if improved else 0.4,
            }
            rationale = "Adjust learning rate based on whether the prior candidate improved the primary metric."
        else:
            best_config = (best or {}).get("config", {})
            changes = {
                "learning_rate": best_config.get("learning_rate", 0.6),
                "discount_factor": 0.99,
                "epsilon_start": best_config.get("epsilon_start", 0.001),
                "epsilon_end": 0.0,
                "epsilon_decay": best_config.get("epsilon_decay", 1.0),
            }
            rationale = "Keep the best learning setup and test a higher discount factor with deterministic late policy."

        return {
            "candidate_id": f"iteration_{iteration:03d}",
            "planner": "rules",
            "rationale": rationale,
            "diagnosis": diagnosis,
            "changes": changes,
            "expected_effect": "May improve pipe-passing behavior under the current training budget.",
            "risk": "Longer budgets cost more wall time; reward remains a diagnostic signal when score is sparse.",
        }


def _improved(last, best, context):
    if not last or not best:
        return False
    metric = context["objective"].get("primary_metric", "mean_score")
    min_improvement = float(context["objective"].get("min_improvement", 0.0))
    return float(last["summary"].get(metric, 0.0)) > float(best["summary"].get(metric, 0.0)) + min_improvement


def _diagnose(history, best):
    summaries = []
    if best:
        summaries.append(best.get("summary", {}))
    summaries.extend(item.get("summary", {}) for item in history)
    scores = [float(s.get("mean_score", 0.0)) for s in summaries if s.get("status") == "success"]
    rewards = [_mean_reward(s) for s in summaries if s.get("status") == "success"]
    return {
        "all_scores_zero": bool(scores) and max(scores) <= 0.0,
        "best_score": max(scores) if scores else None,
        "best_reward": max(rewards) if rewards else None,
        "reason": "score is sparse; use training budget and reward diagnostics" if scores and max(scores) <= 0.0 else "primary score has signal",
    }


def _mean_reward(summary):
    rewards = [
        item.get("eval_result", {}).get("mean_reward", 0.0)
        for item in summary.get("seed_results", [])
        if item.get("eval_status") in {"ok", "success"}
    ]
    return sum(rewards) / len(rewards) if rewards else 0.0


def _next_allowed(context, key, current_value):
    values = sorted(context["search_space"].get(key, [current_value]))
    for value in values:
        if value > current_value:
            return value
    return values[-1]


def _can_increase(context, key, current_value):
    return _next_allowed(context, key, current_value) > current_value
