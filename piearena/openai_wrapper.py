"""Thin client for OpenAI-compatible chat-completion endpoints (e.g. OpenRouter).

Adds retry with exponential backoff on transient errors, a fallback that drops an
unsupported ``response_format``, and defensive extraction of the reply text.

Environment: ``OPENAI_API_KEY`` (or ``OPENROUTER_API_KEY``), ``OPENAI_BASE_URL``,
``OPENAI_HTTP_TIMEOUT_SEC`` (default 120), ``OPENAI_MAX_RETRIES`` (default 3),
``OPENAI_RETRY_BACKOFF`` (default 1.6).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import openai
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError

load_dotenv()

_RETRY_TEXT = ("jsondecodeerror", "expecting value", "parse", "invalid json",
               "missing choices", "choices[0].message")
_SCHEMA_ERROR_TEXT = ("response_format", "json_schema", "Unsupported", "schema")


def _strip_code_fence(s: str) -> str:
    t = (s or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else t
    return t.strip()


def _reply_text(resp: Any) -> str:
    """Return the message text; fall back to tool-call arguments if the text is empty."""
    choices = getattr(resp, "choices", None)
    if not choices or getattr(choices[0], "message", None) is None:
        raise ValueError("Response missing choices[0].message")
    msg = choices[0].message
    text = (getattr(msg, "content", "") or "").strip()
    if text:
        return text
    calls = [{"name": tc.function.name, "arguments": tc.function.arguments}
             for tc in (getattr(msg, "tool_calls", None) or [])]
    fc = getattr(msg, "function_call", None)
    if not calls and fc is not None:
        calls = [{"name": getattr(fc, "name", None), "arguments": getattr(fc, "arguments", None)}]
    if not calls:
        return text
    args = max(calls, key=lambda c: len(c["arguments"] or ""))["arguments"]
    if isinstance(args, str):
        return _strip_code_fence(args)
    if isinstance(args, (dict, list)):
        return json.dumps(args, ensure_ascii=False)
    return args or ""


def _is_transient(exc: Exception) -> bool:
    # Every SDK error (incl. any HTTP status error) and gateway parse glitches are retried.
    if isinstance(exc, (APITimeoutError, APIConnectionError, APIError)):
        return True
    return any(k in str(exc).lower() for k in _RETRY_TEXT)


class OpenAIWrapper:
    """``chat(messages)`` returns ``{"model", "label", "messages", "content"}``."""

    def __init__(self, model: str, label: str | None = None, allow_response_format: bool = False,
                 api_key: str | None = None, base_url: str | None = None, timeout: int | None = None,
                 max_retries: int | None = None, backoff: float | None = None):
        self.model = model
        self.label = label or model
        self.allow_response_format = allow_response_format
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY (or OPENROUTER_API_KEY).")
        self.max_retries = max_retries or int(os.getenv("OPENAI_MAX_RETRIES", "3"))
        self.backoff = backoff or float(os.getenv("OPENAI_RETRY_BACKOFF", "1.6"))
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=timeout or int(os.getenv("OPENAI_HTTP_TIMEOUT_SEC", "120")),
        )

    def _create(self, kwargs: dict) -> str:
        return _reply_text(self.client.chat.completions.with_raw_response.create(**kwargs).parse())

    def chat(self, messages: list[dict], response_format: dict | None = None) -> dict:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if response_format and self.allow_response_format:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_format["schema"],
                                "strict": response_format.get("strict", False)},
            }
        for attempt in range(1, self.max_retries + 1):
            try:
                content = self._create(kwargs)
                break
            except Exception as e:  # noqa: BLE001
                if "response_format" in kwargs and any(s in str(e) for s in _SCHEMA_ERROR_TEXT):
                    kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
                    try:
                        content = self._create(kwargs)
                        break
                    except Exception as e2:  # noqa: BLE001
                        e = e2
                if attempt >= self.max_retries or not _is_transient(e):
                    raise e
                time.sleep(self.backoff ** attempt)
        return {"model": self.model, "label": self.label, "messages": messages, "content": content}
