__all__ = ["AgentPlanner", "main"]


def __getattr__(name):
    if name == "AgentPlanner":
        from agent.planner import AgentPlanner
        return AgentPlanner
    if name == "main":
        from agent.main import main
        return main
    raise AttributeError(name)
