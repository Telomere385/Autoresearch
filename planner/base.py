# 这里定义 planner 子系统的基础类型。
# PlannerError 用于向 Agent 报告可控规划失败，BasePlanner 规定所有 planner 都必须实现 propose(context)。
class PlannerError(Exception):
    pass


class BasePlanner:
    def __init__(self, config):
        self.config = config

    def propose(self, context):
        raise NotImplementedError
