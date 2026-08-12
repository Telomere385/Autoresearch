import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


class PlannerError(RuntimeError):
    pass


def propose(context, config):
    """Ask one configured planner for exactly one next experiment."""
    prompt = json.dumps({
        "role": "You are the planner in a small AutoResearch loop.",
        "task": "Propose exactly one next experiment based on the real results so far.",
        "rules": [
            "Return one JSON object and no markdown.",
            "Only change keys and use values listed in allowed_changes.",
            "Do not edit source code or repeat a tested configuration.",
            "Use previous results and failures when choosing the next experiment.",
        ],
        "schema": {
            "hypothesis": "string",
            "changes": {"allowed.parameter": "allowed value"},
            "reason": "string",
            "expected_effect": "string",
        },
        **context,
    }, indent=2)

    provider = config.get("provider", "openai_compatible")
    if provider == "openai_compatible":
        raw = _call_openai_compatible(prompt, config)
    elif provider == "local_command":
        raw = _call_local_command(prompt, config)
    else:
        raise PlannerError(f"Unsupported planner provider: {provider}")
    return _parse_proposal(raw), prompt, raw


def _parse_proposal(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise PlannerError("Planner response did not contain a JSON object")
    try:
        proposal = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise PlannerError(f"Planner response is not valid JSON: {exc}") from exc

    required = ("hypothesis", "changes", "reason", "expected_effect")
    missing = [key for key in required if key not in proposal]
    if missing:
        raise PlannerError(f"Planner response is missing fields: {', '.join(missing)}")
    if not isinstance(proposal["changes"], dict) or not proposal["changes"]:
        raise PlannerError("Planner changes must be a non-empty object")
    for key in ("hypothesis", "reason", "expected_effect"):
        if not isinstance(proposal[key], str) or not proposal[key].strip():
            raise PlannerError(f"Planner field {key} must be a non-empty string")
    return proposal


def _call_openai_compatible(prompt, config):
    api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise PlannerError(f"Missing API key environment variable: {api_key_env}")

    base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
    body = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(config.get("temperature", 0.2)),
    }
    retries = int(config.get("max_retries", 2))
    timeout = float(config.get("timeout_seconds", 60))
    last_error = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    raise PlannerError(f"Planner API call failed after {retries + 1} attempts: {last_error}")


def _call_local_command(prompt, config):
    command = config.get("command")
    if not isinstance(command, list) or not command:
        raise PlannerError("local_command planner requires a non-empty command list")
    command = [str(part).format(python=sys.executable) for part in command]
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=float(config.get("timeout_seconds", 60)),
        )
    except subprocess.TimeoutExpired as exc:
        raise PlannerError(f"Local planner timed out: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PlannerError(f"Local planner exited with {completed.returncode}: {detail}")
    return completed.stdout
