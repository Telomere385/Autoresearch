from .base import BasePlanner


class RulesPlanner(BasePlanner):
    def propose(self, context):
        iteration = context["iteration"]
        best = context.get("best")
        last = context.get("last")

        if iteration == 1:
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
            "changes": changes,
            "expected_effect": "May improve pipe-passing behavior under the current training budget.",
            "risk": "Short smoke budgets may keep score at zero; reward remains the diagnostic signal.",
        }


def _improved(last, best, context):
    if not last or not best:
        return False
    metric = context["objective"].get("primary_metric", "mean_score")
    min_improvement = float(context["objective"].get("min_improvement", 0.0))
    return float(last["summary"].get(metric, 0.0)) > float(best["summary"].get(metric, 0.0)) + min_improvement
