from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

try:
    from langchain_core.messages import BaseMessage
except Exception:  # pragma: no cover - defensive import guard
    BaseMessage = None

_LOCK = threading.Lock()
_SESSION_ID = (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    + "-"
    + uuid4().hex[:8]
)
_MAX_STRING_LENGTH = 20000


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_trace_logging_enabled() -> bool:
    return os.getenv("NIRMA_TRACE_ENABLED", "1").lower() not in {"0", "false", "no"}


def get_trace_log_dir() -> Path:
    raw_dir = os.getenv("NIRMA_LOG_DIR", "logs")
    path = Path(raw_dir).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return path


def get_trace_log_path() -> Path:
    return get_trace_log_dir() / f"trace-{_SESSION_ID}.jsonl"


def list_trace_log_paths() -> list[Path]:
    log_dir = get_trace_log_dir()
    if not log_dir.exists():
        return []
    return sorted(
        log_dir.glob("trace-*.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )


def get_latest_trace_log_path() -> Path | None:
    paths = list_trace_log_paths()
    if not paths:
        return None
    return paths[-1]


def _truncate(value: str) -> str:
    if len(value) <= _MAX_STRING_LENGTH:
        return value
    return value[: _MAX_STRING_LENGTH - 3] + "..."


def serialize_for_json(value):
    if isinstance(value, BaseModel):
        return serialize_for_json(value.model_dump(mode="json"))
    if BaseMessage is not None and isinstance(value, BaseMessage):
        data = {
            "message_type": value.__class__.__name__,
            "content": serialize_for_json(value.content),
        }
        for attr in [
            "name",
            "tool_calls",
            "invalid_tool_calls",
            "additional_kwargs",
            "response_metadata",
            "usage_metadata",
            "id",
        ]:
            attr_value = getattr(value, attr, None)
            if attr_value not in (None, "", [], {}):
                data[attr] = serialize_for_json(attr_value)
        return data
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_for_json(item) for item in value]
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, Exception):
        return {
            "error_type": value.__class__.__name__,
            "error_message": _truncate(str(value)),
        }
    return _truncate(repr(value))


def log_event(event: str, **payload) -> Path | None:
    if not is_trace_logging_enabled():
        return None

    path = get_trace_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": _SESSION_ID,
        "event": event,
        **{key: serialize_for_json(value) for key, value in payload.items()},
    }
    with _LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_trace_events(path: str | Path | None = None, limit: int | None = None) -> list[dict]:
    if path is not None:
        trace_path = Path(path)
    else:
        current_session_path = get_trace_log_path()
        trace_path = (
            current_session_path
            if current_session_path.exists()
            else get_latest_trace_log_path()
        )
    if trace_path is None:
        return []
    if not trace_path.exists():
        return []

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    if limit is not None:
        lines = lines[-limit:]
    return [json.loads(line) for line in lines if line.strip()]


__all__ = [
    "get_trace_log_dir",
    "get_latest_trace_log_path",
    "get_trace_log_path",
    "is_trace_logging_enabled",
    "list_trace_log_paths",
    "log_event",
    "read_trace_events",
    "serialize_for_json",
]
