import argparse
import json

from autoresearch import run_agent


def main():
    parser = argparse.ArgumentParser(description="Run the Mini AutoResearch agent.")
    parser.add_argument("--config", default="configs/autoresearch.yaml")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    result = run_agent(args.config, args.run_id)
    print(json.dumps(result, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
