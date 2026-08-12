# 这是 planner 包的对外入口。
# 当前工程只暴露 LLMPlanner，Agent 通过 build_planner(config) 获取统一的实验规划器实例。
from .llm import LLMPlanner


def build_planner(config):
    return LLMPlanner(config)
