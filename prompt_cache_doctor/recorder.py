"""Anthropic client wrapper that logs requests + usage to JSONL.

Usage:

    import anthropic
    from prompt_cache_doctor import record

    client = record(anthropic.Anthropic(), log_path="calls.jsonl", route="/chat")
    resp = client.messages.create(model="claude-sonnet-4-6", ...)

The wrapper intercepts `messages.create` (sync) calls, executes them normally,
and appends one JSONL line per call with the request shape + usage block. It
does NOT modify behavior or add latency beyond a single file append.
"""

import json
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional


_log_lock = threading.Lock()


def _serializable(obj: Any) -> Any:
    """Best-effort JSON-safe dump of arbitrary objects (Pydantic, dataclass, etc.)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        try:
            return _serializable(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return _serializable(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _serializable(vars(obj))
        except Exception:
            pass
    return repr(obj)


def _write_log(path: str, payload: dict) -> None:
    line = json.dumps(payload, default=str, ensure_ascii=False)
    with _log_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _wrap_create(original: Callable, log_path: str, route: str) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        response = original(*args, **kwargs)

        usage = {}
        try:
            usage_obj = getattr(response, "usage", None)
            if usage_obj is not None:
                if hasattr(usage_obj, "model_dump"):
                    usage = usage_obj.model_dump()
                else:
                    usage = dict(usage_obj.__dict__)
        except Exception:
            usage = {}

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "route": route,
            "request": {
                "model": kwargs.get("model"),
                "system": _serializable(kwargs.get("system")),
                "messages": _serializable(kwargs.get("messages")),
                "tools": _serializable(kwargs.get("tools")),
            },
            "usage": usage,
        }
        try:
            _write_log(log_path, payload)
        except Exception:
            pass
        return response

    return wrapper


def record(client: Any, log_path: str, route: str = "default") -> Any:
    """Wrap an `anthropic.Anthropic` client so every `messages.create` call is logged.

    Returns the same client object, mutated in place. Safe to call multiple times
    with different log paths on different clients; calls are thread-safe.
    """
    messages_obj = getattr(client, "messages", None)
    if messages_obj is None:
        raise TypeError("Expected an anthropic.Anthropic client (with .messages)")

    original_create = getattr(messages_obj, "create", None)
    if original_create is None:
        raise TypeError("Client has no .messages.create method")

    messages_obj.create = _wrap_create(original_create, log_path=log_path, route=route)
    return client
