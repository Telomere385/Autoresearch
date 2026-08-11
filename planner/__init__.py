from .llm import LLMPlanner


def build_planner(config):
    return LLMPlanner(config)
