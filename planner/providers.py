import json
import os
import subprocess
import urllib.request

from .base import PlannerError


def call_provider(planner_config, prompt):
    provider = planner_config.get("provider", "openai")
    if provider == "openai":
        return call_openai(planner_config, prompt)
    if provider == "local_endpoint":
        return call_local_endpoint(planner_config, prompt)
    if provider == "local_command":
        return call_local_command(planner_config, prompt)
    if provider == "disabled":
        raise PlannerError("LLM provider is disabled.")
    raise PlannerError(f"Unsupported LLM provider: {provider}")


def call_openai(planner_config, prompt):
    api_key_env = planner_config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise PlannerError(f"Missing OpenAI API key environment variable: {api_key_env}")

    model = planner_config.get("model", "gpt-4o-mini")
    timeout = int(planner_config.get("timeout_seconds", 60))
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an experiment planner. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": float(planner_config.get("temperature", 0.2)),
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise PlannerError(f"OpenAI planner call failed: {exc}") from exc
    return data["choices"][0]["message"]["content"]


def call_local_endpoint(planner_config, prompt):
    endpoint = planner_config.get("local_endpoint")
    if not endpoint:
        raise PlannerError("local_endpoint provider requires planner.local_endpoint.")
    timeout = int(planner_config.get("timeout_seconds", 60))
    payload = json.dumps({
        "model": planner_config.get("model"),
        "prompt": prompt,
    }).encode("utf-8")
    req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except Exception as exc:
        raise PlannerError(f"Local endpoint planner call failed: {exc}") from exc
    return unwrap_local_response(body)


def call_local_command(planner_config, prompt):
    command = planner_config.get("local_command")
    if not command:
        raise PlannerError("local_command provider requires planner.local_command.")
    timeout = int(planner_config.get("timeout_seconds", 60))
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


def unwrap_local_response(body):
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body
    if isinstance(data, str):
        return data
    for key in ("candidate", "content", "text", "response", "output"):
        value = data.get(key) if isinstance(data, dict) else None
        if isinstance(value, dict):
            return json.dumps(value)
        if isinstance(value, str):
            return value
    return body
