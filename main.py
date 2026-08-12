import argparse
import json

from autoresearch import run_agent


def main():
    parser = argparse.ArgumentParser(description="Run the Mini AutoResearch agent.")
    parser.add_argument("--config", default="configs/autoresearch.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--goal", default=None, help="Natural-language goal; overrides goal_file")
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--quiet", action="store_true", help="Disable terminal progress")
    progress.add_argument("--verbose", action="store_true", help="Show detailed terminal progress")
    args = parser.parse_args()
    progress_mode = "quiet" if args.quiet else "verbose" if args.verbose else "normal"
    result = run_agent(args.config, args.run_id, args.goal, progress_mode=progress_mode)
    print(json.dumps(result, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
