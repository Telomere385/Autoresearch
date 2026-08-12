import json
import os
import time
import urllib.error
import urllib.request


class PlannerError(RuntimeError):
    pass


def chat(messages, tools, config, tool_choice="required", reporter=None):
    """Call an OpenAI-compatible model and return one assistant message."""
    if config.get("provider", "openai_compatible") != "openai_compatible":
        raise PlannerError("Only the openai_compatible LLM provider is supported")
    api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise PlannerError(f"Missing API key environment variable: {api_key_env}")

    body = {
        "model": config["model"],
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": False,
        "temperature": float(config.get("temperature", 0.2)),
    }
    retries = int(config.get("max_retries", 2))
    timeout = float(config.get("timeout_seconds", 60))
    endpoint = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
    last_error = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            f"{endpoint}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            message = payload["choices"][0]["message"]
            return _normalize_assistant_message(message), payload
        except (
            KeyError,
            IndexError,
            json.JSONDecodeError,
            urllib.error.URLError,
            TimeoutError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                delay = min(2 ** attempt, 8)
                if reporter is not None:
                    reporter.emit(
                        "RETRY",
                        f"LLM API attempt {attempt + 1} failed ({type(exc).__name__}); retrying in {delay}s",
                        verbose=True,
                    )
                time.sleep(delay)
    raise PlannerError(f"LLM API call failed after {retries + 1} attempts: {last_error}")


def _normalize_assistant_message(message):
    if not isinstance(message, dict):
        raise PlannerError("LLM assistant message is not an object")
    normalized = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    # Kimi reasoning models require this field to be replayed verbatim.
    if "reasoning_content" in message:
        normalized["reasoning_content"] = message.get("reasoning_content") or ""
    if message.get("tool_calls") is not None:
        normalized["tool_calls"] = message["tool_calls"]
    return normalized


def parse_tool_call(tool_call):
    try:
        call_id = str(tool_call["id"])
        function = tool_call["function"]
        name = str(function["name"])
        arguments = json.loads(function.get("arguments") or "{}")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PlannerError(f"Invalid tool call: {exc}") from exc
    if not call_id or not name or not isinstance(arguments, dict):
        raise PlannerError("Tool call id, name, and object arguments are required")
    return call_id, name, arguments
