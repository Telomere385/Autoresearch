"""State-dependent offline stand-in for smoke-testing the planner protocol."""

import json
import sys


def main():
    request = json.loads(sys.stdin.read())
    history = request.get("previous_experiments", [])
    current = request.get("current_config", {})

    if not history:
        proposal = {
            "hypothesis": "More exploration may expose useful actions during the short training run.",
            "changes": {"epsilon_start": 0.05},
            "reason": "No candidate has been tested yet, so start with one bounded exploration change.",
            "expected_effect": "The learned table may follow a different evaluation trajectory.",
        }
    elif history[-1]["decision"] == "accept":
        proposal = {
            "hypothesis": "A lower discount may prefer immediate survival signals after the accepted result.",
            "changes": {"discount_factor": 0.9},
            "reason": "The previous real result was accepted; test whether shorter-horizon updates improve it further.",
            "expected_effect": "Immediate rewards should influence Q updates more strongly.",
        }
    elif current.get("rewards", {}).get("alive") != 0.1:
        proposal = {
            "hypothesis": "A denser survival reward may help after the previous candidate was rejected.",
            "changes": {"rewards.alive": 0.1},
            "reason": "The previous real result did not improve, so switch from value horizon to reward shaping.",
            "expected_effect": "Mean reward may improve even when the short-run score remains tied.",
        }
    else:
        proposal = {
            "hypothesis": "A shorter action sampling interval may refine control after reward shaping.",
            "changes": {"sample_t": 2},
            "reason": "Reward shaping is already active, so test a distinct control-frequency hypothesis.",
            "expected_effect": "More frequent decisions may change survival and score.",
        }
    print(json.dumps(proposal))


if __name__ == "__main__":
    main()
