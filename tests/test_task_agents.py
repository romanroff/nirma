from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage

from src import AgentTask, DocumentResearchAgent, WebResearchAgent, WorkerResponse


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


def _web_result(url: str, title: str, snippet: str, source_type: str = "web") -> dict:
    return {
        "type": source_type,
        "title": title,
        "locator": url,
        "snippet": snippet,
        "metadata": {},
    }


def _build_search_backend(mapping: dict[tuple[str, str], list[dict]]):
    def search_backend(query: str, *, source: str, max_results: int) -> dict:
        return {"results": list(mapping.get((source, query), []))[:max_results]}

    return search_backend


def _build_page_fetcher(
    payloads: dict[str, dict],
    calls: list[str] | None = None,
):
    def page_fetcher(url: str, *, timeout: int) -> dict:
        del timeout
        if calls is not None:
            calls.append(url)
        return dict(
            payloads.get(
                url,
                {
                    "requested_url": url,
                    "final_url": url,
                    "status_code": 404,
                    "content_type": "text/html",
                    "html": "",
                    "error": "http_404",
                },
            )
        )

    return page_fetcher


def _build_content_extractor(mapping: dict[str, str]):
    def content_extractor(url: str, html: str) -> str:
        del html
        return mapping.get(url, "")

    return content_extractor


def _build_web_worker(**kwargs) -> WebResearchAgent:
    worker = WebResearchAgent(model=FakeChatModel(), **kwargs)
    worker._plan_search_queries = lambda task: [
        "gatchina transport strategy",
        "gatchina tourism development",
        "gatchina economy growth",
    ]
    worker._synthesize_summary = (
        lambda task, passages: (
            f"Summary based on {len(passages)} evidence passages."
            if passages
            else ""
        )
    )
    return worker


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


def test_web_agent_executes_perplexity_like_pipeline():
    task = AgentTask(kind="web_research", query="How should Gatchina grow?")
    queries = [
        "gatchina transport strategy",
        "gatchina tourism development",
        "gatchina economy growth",
    ]
    urls = [
        "https://alpha.example/transport",
        "https://bravo.example/tourism",
        "https://charlie.example/economy",
        "https://delta.example/planning",
        "https://echo.example/investment",
        "https://foxtrot.example/mobility",
    ]
    search_mapping = {
        ("duckduckgo", queries[0]): [
            _web_result(urls[0], "Transport update", "Search snippet alpha"),
            _web_result(urls[1], "Tourism update", "Search snippet bravo"),
        ],
        ("duckduckgo", queries[1]): [
            _web_result(urls[2], "Economy update", "Search snippet charlie"),
            _web_result(urls[3], "Planning update", "Search snippet delta"),
        ],
        ("duckduckgo", queries[2]): [
            _web_result(urls[4], "Investment update", "Search snippet echo"),
            _web_result(urls[5], "Mobility update", "Search snippet foxtrot"),
        ],
    }
    fetch_mapping = {
        url: {
            "requested_url": url,
            "final_url": url,
            "status_code": 200,
            "content_type": "text/html; charset=utf-8",
            "html": f"<html><body>{url}</body></html>",
            "error": None,
        }
        for url in urls
    }
    extractor_mapping = {
        url: (
            f"{name} evidence about Gatchina transport, tourism, economy and "
            "investment priorities. This page contains concrete strategic details. "
            "The city should improve transit, visitor infrastructure, jobs, and "
            "public space. "
        )
        * 12
        for url, name in zip(
            urls,
            ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"],
        )
    }
    worker = _build_web_worker(
        search_backend=_build_search_backend(search_mapping),
        page_fetcher=_build_page_fetcher(fetch_mapping),
        content_extractor=_build_content_extractor(extractor_mapping),
        wiki_enabled=False,
    )

    result = worker.execute(task)

    assert result.task_id == task.task_id
    assert result.agent_name == "web_research_agent"
    assert result.status == "success"
    assert result.summary == "Summary based on 12 evidence passages."
    assert len(result.sources) == 6
    assert all(source.locator in urls for source in result.sources)
    assert "Search snippet" not in result.sources[0].snippet
    assert result.artifacts["counts"]["page_count"] == 6
    assert result.artifacts["counts"]["used_passage_count"] == 12
    assert result.error is None


def test_web_agent_enforces_dedup_and_per_domain_cap():
    task = AgentTask(kind="web_research", query="Find facts about Gatchina.")
    calls: list[str] = []
    query = "gatchina transport strategy"
    search_mapping = {
        ("duckduckgo", query): [
            _web_result("https://same.example/a", "Same A", "same snippet"),
            _web_result("https://same.example/a?utm_source=x", "Same A dup", "dup"),
            _web_result("https://same.example/b", "Same B", "same snippet"),
            _web_result("https://same.example/c", "Same C", "same snippet"),
            _web_result("https://other.example/d", "Other D", "other snippet"),
        ],
        ("duckduckgo", "gatchina tourism development"): [],
        ("duckduckgo", "gatchina economy growth"): [],
    }
    fetch_mapping = {
        "https://same.example/a": {
            "requested_url": "https://same.example/a",
            "final_url": "https://same.example/a",
            "status_code": 200,
            "content_type": "text/html",
            "html": "<html>a</html>",
            "error": None,
        },
        "https://same.example/b": {
            "requested_url": "https://same.example/b",
            "final_url": "https://same.example/b",
            "status_code": 200,
            "content_type": "text/html",
            "html": "<html>b</html>",
            "error": None,
        },
        "https://other.example/d": {
            "requested_url": "https://other.example/d",
            "final_url": "https://other.example/d",
            "status_code": 200,
            "content_type": "text/html",
            "html": "<html>d</html>",
            "error": None,
        },
    }
    extractor_mapping = {
        "https://same.example/a": "A evidence about Gatchina strategy. " * 20,
        "https://same.example/b": "B evidence about Gatchina strategy. " * 20,
        "https://other.example/d": "D evidence about Gatchina strategy. " * 20,
    }
    worker = _build_web_worker(
        search_backend=_build_search_backend(search_mapping),
        page_fetcher=_build_page_fetcher(fetch_mapping, calls=calls),
        content_extractor=_build_content_extractor(extractor_mapping),
        wiki_enabled=False,
        fetch_budget=5,
        per_domain_cap=2,
    )

    result = worker.execute(task)

    assert result.status == "partial"
    assert calls == [
        "https://same.example/a",
        "https://same.example/b",
        "https://other.example/d",
    ]
    assert len(
        [
            candidate
            for candidate in result.artifacts["candidate_urls"]
            if candidate["domain"] == "same.example"
        ]
    ) == 2


def test_web_agent_returns_partial_when_only_some_pages_are_usable():
    task = AgentTask(kind="web_research", query="How should Gatchina grow?")
    queries = [
        "gatchina transport strategy",
        "gatchina tourism development",
        "gatchina economy growth",
    ]
    search_mapping = {
        ("duckduckgo", queries[0]): [
            _web_result("https://alpha.example/a", "Alpha", "alpha snippet"),
            _web_result("https://bravo.example/b", "Bravo", "bravo snippet"),
        ],
        ("duckduckgo", queries[1]): [
            _web_result("https://charlie.example/c", "Charlie", "charlie snippet"),
            _web_result("https://delta.example/d", "Delta", "delta snippet"),
        ],
        ("duckduckgo", queries[2]): [
            _web_result("https://echo.example/e", "Echo", "echo snippet"),
        ],
    }
    fetch_mapping = {
        "https://alpha.example/a": {
            "requested_url": "https://alpha.example/a",
            "final_url": "https://alpha.example/a",
            "status_code": 200,
            "content_type": "text/html",
            "html": "<html>a</html>",
            "error": None,
        },
        "https://bravo.example/b": {
            "requested_url": "https://bravo.example/b",
            "final_url": "https://bravo.example/b",
            "status_code": 200,
            "content_type": "application/pdf",
            "html": "",
            "error": "non_html_content_type",
        },
        "https://charlie.example/c": {
            "requested_url": "https://charlie.example/c",
            "final_url": "https://charlie.example/c",
            "status_code": 200,
            "content_type": "text/html",
            "html": "<html>c</html>",
            "error": None,
        },
        "https://delta.example/d": {
            "requested_url": "https://delta.example/d",
            "final_url": "https://delta.example/d",
            "status_code": None,
            "content_type": "",
            "html": "",
            "error": "timeout",
        },
        "https://echo.example/e": {
            "requested_url": "https://echo.example/e",
            "final_url": "https://echo.example/e",
            "status_code": 200,
            "content_type": "text/html",
            "html": "<html>e</html>",
            "error": None,
        },
    }
    extractor_mapping = {
        "https://alpha.example/a": "Alpha evidence about Gatchina transport and growth. " * 20,
        "https://charlie.example/c": "Charlie evidence about Gatchina tourism and growth. " * 20,
        "https://echo.example/e": "Echo evidence about Gatchina economy and growth. " * 20,
    }
    worker = _build_web_worker(
        search_backend=_build_search_backend(search_mapping),
        page_fetcher=_build_page_fetcher(fetch_mapping),
        content_extractor=_build_content_extractor(extractor_mapping),
        wiki_enabled=False,
    )

    result = worker.execute(task)

    assert result.status == "partial"
    assert result.summary == "Summary based on 5 evidence passages."
    assert result.artifacts["counts"]["page_count"] == 3
    skipped_reasons = {
        item["reason"] for item in result.artifacts["skipped_urls"]
    }
    assert "non_html_content_type" in skipped_reasons
    assert "timeout" in skipped_reasons


def test_web_agent_returns_failed_when_no_usable_evidence_exists():
    task = AgentTask(kind="web_research", query="How should Gatchina grow?")
    queries = [
        "gatchina transport strategy",
        "gatchina tourism development",
        "gatchina economy growth",
    ]
    search_mapping = {
        ("duckduckgo", queries[0]): [
            _web_result("https://alpha.example/a", "Alpha", ""),
        ],
        ("duckduckgo", queries[1]): [
            _web_result("https://bravo.example/b", "Bravo", ""),
        ],
        ("duckduckgo", queries[2]): [],
    }
    fetch_mapping = {
        "https://alpha.example/a": {
            "requested_url": "https://alpha.example/a",
            "final_url": "https://alpha.example/a",
            "status_code": None,
            "content_type": "",
            "html": "",
            "error": "timeout",
        },
        "https://bravo.example/b": {
            "requested_url": "https://bravo.example/b",
            "final_url": "https://bravo.example/b",
            "status_code": 500,
            "content_type": "text/html",
            "html": "",
            "error": "http_500",
        },
    }
    worker = _build_web_worker(
        search_backend=_build_search_backend(search_mapping),
        page_fetcher=_build_page_fetcher(fetch_mapping),
        content_extractor=_build_content_extractor({}),
        wiki_enabled=False,
    )

    result = worker.execute(task)

    assert result.status == "failed"
    assert result.summary == ""
    assert result.sources == []
    assert result.error == "Agent reported that it could not complete the task."


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
