import json

from .base import BasePlanner, PlannerError
from .providers import call_provider
from .rules import RulesPlanner


class LLMPlanner(BasePlanner):
    def __init__(self, config):
        super().__init__(config)
        self.fallback = RulesPlanner(config)

    def propose(self, context):
        prompt = _build_prompt(context)
        try:
            raw = call_provider(self.config.get("planner", {}), prompt)
            candidate = json.loads(raw)
            if not isinstance(candidate, dict) or not isinstance(candidate.get("changes"), dict):
                raise PlannerError("LLM output must be a JSON object with a changes object.")
            candidate.setdefault("planner", "llm")
            candidate.setdefault("candidate_id", f"iteration_{context['iteration']:03d}")
            return candidate
        except Exception as exc:
            candidate = self.fallback.propose(context)
            candidate["planner"] = "rules_fallback"
            candidate["fallback_reason"] = str(exc)
            return candidate


def _build_prompt(context):
    payload = {
        "goal": context.get("goal", ""),
        "iteration": context.get("iteration"),
        "objective": context.get("objective", {}),
        "search_space": context.get("search_space", {}),
        "constraints": context.get("constraints", {}),
        "current_config": context.get("current_config", {}),
        "best": context.get("best"),
        "last": context.get("last"),
        "history": context.get("history", [])[-3:],
        "instruction": "Return only JSON with candidate_id, rationale, changes, expected_effect, and risk.",
    }
    return json.dumps(payload, indent=2)
