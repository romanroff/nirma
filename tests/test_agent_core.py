import json

from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src import Agent
from src import Role
from src import RoleAgent
from src.board import BaseNote
from src.agents.factories.decider import DeciderResponse


class DemoResponse(BaseModel):
    value: str = Field(description="Demo value")


def test_agent_parses_structured_json_from_ai_message():
    agent = Agent(
        model=FakeChatModel(),
        system_prompt="Return a structured response.",
        response_format=DemoResponse,
    )
    agent._agent.invoke = lambda *args, **kwargs: {
        "messages": [AIMessage(content='{"value":"ok"}')]
    }

    result = agent.invoke([])

    assert isinstance(result, DemoResponse)
    assert result.value == "ok"


def test_agent_parses_json_inside_code_fence():
    agent = Agent(
        model=FakeChatModel(),
        system_prompt="Return a structured response.",
        response_format=DemoResponse,
    )
    agent._agent.invoke = lambda *args, **kwargs: {
        "messages": [AIMessage(content='```json\n{"value":"ok"}\n```')]
    }

    result = agent.invoke([])

    assert result.value == "ok"


def test_role_agent_preserves_structured_output_instructions():
    agent = RoleAgent(
        role=Role(name="Планировщик", description="Готовит план."),
        model=FakeChatModel(),
        system_prompt="Вы {role_name}.",
        response_format=DemoResponse,
    )

    assert "Вы Планировщик." in agent.system_prompt
    assert "Верни только валидный JSON." in agent.system_prompt
    assert "{role_name}" not in agent.system_prompt


def test_agent_repairs_overlong_base_note_fields():
    agent = Agent(
        model=FakeChatModel(),
        system_prompt="Return a structured response.",
        response_format=BaseNote,
    )
    agent._agent.invoke = lambda *args, **kwargs: {
        "messages": [
            AIMessage(
                content=json.dumps(
                    {
                        "content": "x" * 2105,
                        "summary": "y" * 320,
                        "keywords": [f"tag-{i}" for i in range(7)],
                    },
                    ensure_ascii=False,
                )
            )
        ]
    }

    result = agent.invoke([])

    assert len(result.content) <= 2000
    assert result.content.endswith("...")
    assert len(result.summary) <= 280
    assert result.summary.endswith("...")
    assert len(result.keywords) == 5


def test_agent_repairs_nested_base_note_fields():
    agent = Agent(
        model=FakeChatModel(),
        system_prompt="Return a structured response.",
        response_format=DeciderResponse,
    )
    agent._agent.invoke = lambda *args, **kwargs: {
        "messages": [
            AIMessage(
                content=json.dumps(
                    {
                        "note": {
                            "content": "x" * 2200,
                            "summary": "y" * 400,
                            "keywords": [f"tag-{i}" for i in range(8)],
                        },
                        "is_final": False,
                    },
                    ensure_ascii=False,
                )
            )
        ]
    }

    result = agent.invoke([])

    assert len(result.note.content) <= 2000
    assert len(result.note.summary) <= 280
    assert len(result.note.keywords) == 5
