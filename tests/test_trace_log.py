import importlib

from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

import src.trace_log as trace_log
from src import Agent


class LoggedResponse(BaseModel):
    value: str = Field(description="Logged value")


def test_trace_log_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("NIRMA_LOG_DIR", str(tmp_path))

    trace_log.log_event("demo_event", payload={"value": "ok"})
    events = trace_log.read_trace_events(trace_log.get_trace_log_path())

    assert events[-1]["event"] == "demo_event"
    assert events[-1]["payload"]["value"] == "ok"


def test_trace_log_lists_and_reads_latest_existing_log(tmp_path, monkeypatch):
    monkeypatch.setenv("NIRMA_LOG_DIR", str(tmp_path))

    older = tmp_path / "trace-20260325T141227Z-older.jsonl"
    newer = tmp_path / "trace-20260325T141323Z-newer.jsonl"
    older.write_text('{"event":"older"}\n', encoding="utf-8")
    newer.write_text('{"event":"newer"}\n', encoding="utf-8")

    paths = trace_log.list_trace_log_paths()
    latest = trace_log.get_latest_trace_log_path()
    events = trace_log.read_trace_events()

    assert [path.name for path in paths] == [older.name, newer.name]
    assert latest == newer
    assert events[-1]["event"] == "newer"


def test_agent_invoke_emits_trace_events(monkeypatch):
    events = []
    monkeypatch.setattr(
        "src.agents.agent.trace_log.log_event",
        lambda event, **payload: events.append((event, payload)),
    )

    agent = Agent(
        model=FakeChatModel(),
        system_prompt="Return JSON.",
        response_format=LoggedResponse,
    )
    agent._agent.invoke = lambda *args, **kwargs: {
        "messages": [AIMessage(content='{"value":"ok"}')]
    }

    result = agent.invoke([])

    assert result.value == "ok"
    assert [event for event, _ in events] == [
        "agent_invoke_start",
        "agent_invoke_raw_response",
        "agent_invoke_result",
    ]
