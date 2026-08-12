"""OpenAI-compatible planner transport and native tool-call parsing."""

import json
import os

import openai
from openai import OpenAI


class PlannerError(RuntimeError):
    """Represent a recoverable planner request or response error."""


def chat(messages, tools, config, tool_choice="required", reporter=None, client_factory=OpenAI):
    """Call an OpenAI-compatible model through the official Python SDK.

    The returned assistant message is normalized for replay, while the raw SDK
    payload is retained so the agent can persist a complete execution record.
    """
    if config.get("provider", "openai_compatible") != "openai_compatible":
        raise PlannerError("Only the openai_compatible LLM provider is supported")
    api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise PlannerError(f"Missing API key environment variable: {api_key_env}")

    timeout = float(config.get("timeout_seconds", 60))
    retries = int(config.get("max_retries", 2))
    base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
    try:
        client = client_factory(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=retries,
        )
        completion = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=False,
            temperature=float(config.get("temperature", 0.2)),
        )
        payload = completion.model_dump(mode="json")
        message = payload["choices"][0]["message"]
        return _normalize_assistant_message(message), payload
    except openai.APIError as exc:
        detail = _sdk_error_detail(exc)
        if reporter is not None:
            reporter.emit("ERROR", f"LLM SDK request failed: {detail}", verbose=True)
        raise PlannerError(f"LLM SDK request failed: {detail}") from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PlannerError(f"LLM SDK returned an invalid response: {exc}") from exc


def _sdk_error_detail(exc):
    """Build a bounded diagnostic string without exposing request secrets."""
    parts = [type(exc).__name__]
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    if status_code is not None:
        parts.append(f"status={status_code}")
    if request_id:
        parts.append(f"request_id={request_id}")
    message = str(exc).strip()
    if message:
        parts.append(message[:1000])
    return "; ".join(parts)


def _normalize_assistant_message(message):
    """Preserve the assistant fields required by subsequent chat turns."""
    if not isinstance(message, dict):
        raise PlannerError("LLM assistant message is not an object")
    normalized = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    # DashScope Kimi reasoning models require this provider-specific field to
    # be replayed verbatim in later assistant messages. The SDK retains unknown
    # response properties when model_dump() serializes the response model.
    if "reasoning_content" in message:
        normalized["reasoning_content"] = message.get("reasoning_content") or ""
    if message.get("tool_calls") is not None:
        normalized["tool_calls"] = message["tool_calls"]
    return normalized


def parse_tool_call(tool_call):
    """Decode one native function call into its id, name, and argument mapping."""
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
