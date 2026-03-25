from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage

from src import (
    AgentSource,
    AgentTask,
    DocumentResearchAgent,
    WebResearchAgent,
    WorkerResponse,
)


@tool
def fake_web_search(query: str) -> dict:
    """Return fake web results."""
    return {
        "results": [
            {
                "type": "web",
                "title": "Example",
                "locator": "https://example.com",
                "snippet": "Example snippet",
                "metadata": {},
            }
        ]
    }


@tool
def fake_wiki_search(query: str) -> dict:
    """Return fake wiki results."""
    return {
        "results": [
            {
                "type": "wikipedia",
                "title": "Example Wiki",
                "locator": "https://ru.wikipedia.org/wiki/Example",
                "snippet": "Example wiki snippet",
                "metadata": {},
            }
        ]
    }


@tool
def fake_document_search(query: str) -> dict:
    """Return fake document results."""
    return {
        "results": [
            {
                "type": "document",
                "title": "plan.pdf",
                "locator": "plan.pdf#page=2&chunk=5",
                "snippet": "Document snippet",
                "metadata": {"page_number": 2, "chunk_index": 5},
            }
        ]
    }


class FakeStore:
    research_tool = fake_document_search


def test_web_agent_builds_default_tools():
    worker = WebResearchAgent(model=FakeChatModel())

    assert worker.name == "web_research_agent"
    assert worker.capabilities == ("web_research",)
    assert {tool.name for tool in worker.tools} == {
        "duckduckgo_search",
        "wikipedia_search",
    }


def test_document_agent_uses_store_tool():
    worker = DocumentResearchAgent(model=FakeChatModel(), store=FakeStore())

    assert worker.name == "document_research_agent"
    assert worker.capabilities == ("document_research",)
    assert [tool.name for tool in worker.tools] == ["fake_document_search"]


def test_supports_routes_by_task_kind():
    web_worker = WebResearchAgent(
        model=FakeChatModel(),
        tools=[fake_web_search, fake_wiki_search],
    )
    document_worker = DocumentResearchAgent(
        model=FakeChatModel(),
        tool=fake_document_search,
    )

    web_task = AgentTask(kind="web_research", query="Find facts about Gatchina.")
    document_task = AgentTask(
        kind="document_research",
        query="Find priorities in the local plan.",
    )

    assert web_worker.supports(web_task) is True
    assert web_worker.supports(document_task) is False
    assert document_worker.supports(document_task) is True
    assert document_worker.supports(web_task) is False


def test_execute_returns_normalized_agent_result():
    worker = WebResearchAgent(
        model=FakeChatModel(),
        tools=[fake_web_search, fake_wiki_search],
    )
    task = AgentTask(kind="web_research", query="Find facts about Gatchina.")

    worker._agent.stream = lambda *args, **kwargs: iter(
        [
            {
                "model": {
                    "messages": [
                        AIMessage(
                            content=WorkerResponse(
                                status="success",
                                summary="Gatchina can focus on mobility and tourism.",
                                sources=[
                                    AgentSource(
                                        type="web",
                                        title="Example",
                                        locator="https://example.com",
                                        snippet="Example snippet",
                                        metadata={},
                                    )
                                ],
                            ).model_dump_json()
                        )
                    ]
                }
            }
        ]
    )

    result = worker.execute(task)

    assert result.task_id == task.task_id
    assert result.agent_name == "web_research_agent"
    assert result.status == "success"
    assert result.summary
    assert len(result.sources) == 1
    assert result.error is None


def test_execute_downgrades_source_less_success_to_partial():
    worker = DocumentResearchAgent(
        model=FakeChatModel(),
        tool=fake_document_search,
    )
    task = AgentTask(
        kind="document_research",
        query="Find priorities in the local plan.",
    )

    worker._agent.invoke = lambda *args, **kwargs: {
        "structured_response": WorkerResponse(
            status="success",
            summary="The document mentions priorities but the evidence is weak.",
            sources=[],
        )
    }

    result = worker.execute(task)

    assert result.status == "partial"
    assert result.error is None


def test_execute_returns_failed_result_on_runtime_error():
    worker = DocumentResearchAgent(
        model=FakeChatModel(),
        tool=fake_document_search,
    )
    task = AgentTask(
        kind="document_research",
        query="Find priorities in the local plan.",
    )

    def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    worker._agent.invoke = raise_error

    result = worker.execute(task)

    assert result.status == "failed"
    assert result.error == "boom"


def test_execute_salvages_plain_text_response_as_partial():
    worker = DocumentResearchAgent(
        model=FakeChatModel(),
        tool=fake_document_search,
    )
    task = AgentTask(
        kind="document_research",
        query="Find priorities in the local plan.",
    )

    worker._agent.invoke = lambda *args, **kwargs: {
        "messages": [
            AIMessage(
                content=(
                    "I’m ready to help with your document-based research request. "
                    "Please provide the specific question or information you need."
                )
            )
        ]
    }

    result = worker.execute(task)

    assert result.status == "partial"
    assert "document-based research request" in result.summary
    assert result.sources == []
    assert result.error is None
