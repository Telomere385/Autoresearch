import json
import os
import subprocess
import urllib.request

from .base import PlannerError


def call_provider(planner_config, prompt):
    provider = planner_config.get("provider", "disabled")
    if provider == "disabled":
        raise PlannerError("LLM provider is disabled.")
    if provider == "local":
        return _call_local(planner_config, prompt)
    if provider in {"openai", "anthropic"}:
        return _call_remote_stub(planner_config, provider)
    raise PlannerError(f"Unsupported planner provider: {provider}")


def _call_local(planner_config, prompt):
    endpoint = planner_config.get("local_endpoint")
    command = planner_config.get("local_command")
    timeout = int(planner_config.get("timeout_seconds", 60))
    if endpoint:
        payload = json.dumps({"prompt": prompt}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode("utf-8")
    if command:
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            shell=True,
        )
        if completed.returncode != 0:
            raise PlannerError(completed.stderr.strip() or "Local planner command failed.")
        return completed.stdout
    raise PlannerError("Local planner requires local_endpoint or local_command.")


def _call_remote_stub(planner_config, provider):
    api_key_env = planner_config.get("api_key_env")
    if not api_key_env or not os.environ.get(api_key_env):
        raise PlannerError(f"{provider} planner requires API key env var {api_key_env!r}.")
    raise PlannerError(f"{provider} provider adapter is configured but not enabled in this offline build.")
