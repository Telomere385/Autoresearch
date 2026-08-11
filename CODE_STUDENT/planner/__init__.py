from .rules import RulesPlanner
from .llm import LLMPlanner


def build_planner(config):
    planner_config = config.get("planner", {})
    planner_type = planner_config.get("type", "rules")
    provider = planner_config.get("provider", "disabled")
    if planner_type == "llm" and provider != "disabled":
        return LLMPlanner(config)
    return RulesPlanner(config)
