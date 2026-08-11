class PlannerError(Exception):
    pass


class BasePlanner:
    def __init__(self, config):
        self.config = config

    def propose(self, context):
        raise NotImplementedError
