# 这里实现基于 LLM 的实验规划器。
# 它把 Agent 传入的 context 压缩成 JSON prompt，要求模型只返回候选参数变更，并负责解析和校验返回结构。
import json

from .base import BasePlanner, PlannerError
from .providers import call_provider


class LLMPlanner(BasePlanner):
    def propose(self, context):
        raw = call_provider(self.config.get("planner", {}), build_prompt(context))
        candidate = parse_candidate(raw)
        candidate.setdefault("planner", "llm")
        candidate.setdefault("candidate_id", f"iteration_{context['iteration']:03d}")
        return candidate


def parse_candidate(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise PlannerError("LLM response did not contain a JSON object.")
    candidate = json.loads(text[start:end + 1])
    if not isinstance(candidate, dict):
        raise PlannerError("LLM candidate must be a JSON object.")
    if not isinstance(candidate.get("changes"), dict):
        raise PlannerError("LLM candidate must contain a changes object.")
    return candidate


def build_prompt(context):
    compact_history = []
    for item in context.get("history", [])[-5:]:
        compact_history.append({
            "iteration": item.get("iteration"),
            "decision": item.get("decision"),
            "reason": item.get("reason"),
            "changes": item.get("candidate", {}).get("changes", {}),
            "mean_score": item.get("summary", {}).get("mean_score"),
            "mean_reward": item.get("summary", {}).get("mean_reward"),
        })

    payload = {
        "role": "You are the planner inside a minimal AutoResearch loop.",
        "task": "Propose exactly one next experiment for the configured RL task interface.",
        "hard_rules": [
            "Return only one JSON object. No markdown.",
            "Only propose keys listed in search_space.",
            "Only use values listed in search_space.",
            "Do not propose source edits.",
            "Do not repeat a previous or current configuration.",
            "Prefer small, testable changes with a clear rationale.",
        ],
        "required_output_schema": {
            "candidate_id": "string",
            "rationale": "string",
            "changes": {"allowed_key": "allowed_value"},
            "expected_effect": "string",
            "risk": "string",
        },
        "goal": context.get("goal", ""),
        "program": context.get("program", ""),
        "task_interface": context.get("task", {}),
        "iteration": context.get("iteration"),
        "objective": context.get("objective", {}),
        "search_space": context.get("search_space", {}),
        "constraints": context.get("constraints", {}),
        "current_config": context.get("current_config", {}),
        "best": summarize_result(context.get("best")),
        "last": summarize_result(context.get("last")),
        "history": compact_history,
    }
    return json.dumps(payload, indent=2)


def summarize_result(item):
    if not item:
        return None
    summary = item.get("summary", {})
    return {
        "label": item.get("label"),
        "config": item.get("config", {}),
        "status": summary.get("status"),
        "mean_score": summary.get("mean_score"),
        "mean_reward": summary.get("mean_reward"),
        "max_score": summary.get("max_score"),
    }
