# 这是本地 mock planner。
# 它从 stdin 接收与真实 LLM 相同的 prompt，并返回固定格式的候选配置，用于无网络验证 Agent planning 接口。
import json
import sys


def main():
    context = json.loads(sys.stdin.read())
    iteration = int(context.get("iteration", 1))
    candidates = [
        {
            "candidate_id": "iteration_001",
            "rationale": "Smoke test candidate from local LLM-compatible command.",
            "changes": {"epsilon_start": 0.05, "epsilon_end": 0.001, "epsilon_decay": 0.995},
            "expected_effect": "Exercises the LLM planner path without a remote API call.",
            "risk": "Smoke budgets are too short for meaningful score improvements.",
        },
        {
            "candidate_id": "iteration_002",
            "rationale": "Try a longer-term value estimate.",
            "changes": {"discount_factor": 0.99, "epsilon_start": 0.01, "epsilon_end": 0.0},
            "expected_effect": "May improve survival after more training.",
            "risk": "May not help under tiny smoke budgets.",
        },
        {
            "candidate_id": "iteration_003",
            "rationale": "Try a denser control interval.",
            "changes": {"sample_t": 2, "rewards.alive": 0.1},
            "expected_effect": "May increase reward signal density.",
            "risk": "Changes policy timing and can reduce stability.",
        },
    ]
    print(json.dumps(candidates[min(iteration, len(candidates)) - 1]))


if __name__ == "__main__":
    main()
