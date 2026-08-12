# 这里封装 planner 调用外部或本地模型的 provider 适配层。
# OpenAI-compatible、Anthropic、本地 HTTP endpoint 和本地命令都在这里统一处理请求、重试和错误包装。
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

from .base import PlannerError


def call_provider(planner_config, prompt):
    provider = planner_config.get("provider", "openai")
    if provider == "openai":
        return call_openai(planner_config, prompt)
    if provider == "anthropic":
        return call_anthropic(planner_config, prompt)
    if provider == "local":
        if planner_config.get("local_command"):
            return call_local_command(planner_config, prompt)
        return call_local_endpoint(planner_config, prompt)
    if provider == "local_endpoint":
        return call_local_endpoint(planner_config, prompt)
    if provider == "local_command":
        return call_local_command(planner_config, prompt)
    if provider == "disabled":
        raise PlannerError("LLM provider is disabled.")
    raise PlannerError(f"Unsupported LLM provider: {provider}")


def post_json(url, payload, headers, timeout, max_retries=0, retry_initial_seconds=2.0, error_label="HTTP JSON call"):
    last_error = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code} {exc.reason}: {body or exc}"
            if exc.code != 429 or attempt >= max_retries:
                raise PlannerError(f"{error_label} failed: {last_error}") from exc
            retry_after = exc.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_seconds = float(retry_after)
                except ValueError:
                    sleep_seconds = retry_initial_seconds * (2 ** attempt)
            else:
                sleep_seconds = retry_initial_seconds * (2 ** attempt)
            time.sleep(sleep_seconds)
        except Exception as exc:
            raise PlannerError(f"{error_label} failed: {exc}") from exc
    raise PlannerError(f"{error_label} failed: {last_error}")


def call_openai(planner_config, prompt):
    api_key_env = planner_config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise PlannerError(f"Missing OpenAI API key environment variable: {api_key_env}")

    model = planner_config.get("model", "gpt-4o-mini")
    timeout = int(planner_config.get("timeout_seconds", 60))
    base_url = planner_config.get("base_url", "https://api.openai.com/v1").rstrip("/")
    max_retries = int(planner_config.get("max_retries", 3))
    retry_initial_seconds = float(planner_config.get("retry_initial_seconds", 2.0))
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
    data = post_json(
        f"{base_url}/chat/completions",
        payload,
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout,
        max_retries=max_retries,
        retry_initial_seconds=retry_initial_seconds,
        error_label="OpenAI-compatible planner call",
    )
    return data["choices"][0]["message"]["content"]


def call_anthropic(planner_config, prompt):
    api_key_env = planner_config.get("api_key_env", "ANTHROPIC_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise PlannerError(f"Missing Anthropic API key environment variable: {api_key_env}")

    model = planner_config.get("model", "claude-3-5-haiku-latest")
    timeout = int(planner_config.get("timeout_seconds", 60))
    max_retries = int(planner_config.get("max_retries", 3))
    retry_initial_seconds = float(planner_config.get("retry_initial_seconds", 2.0))
    payload = {
        "model": model,
        "max_tokens": int(planner_config.get("max_tokens", 4096)),
        "temperature": float(planner_config.get("temperature", 0.2)),
        "system": "You are an experiment planner. Return only valid JSON.",
        "messages": [{"role": "user", "content": prompt}],
    }
    data = post_json(
        "https://api.anthropic.com/v1/messages",
        payload,
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        timeout,
        max_retries=max_retries,
        retry_initial_seconds=retry_initial_seconds,
        error_label="Anthropic planner call",
    )
    blocks = data.get("content", [])
    text_blocks = [block.get("text", "") for block in blocks if block.get("type") == "text"]
    return "\n".join(text_blocks)


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
