from __future__ import annotations

import json
import re
from typing import Any, Iterable, Sequence

from langchain.tools import tool
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from ..llms import llm
from .runtime import (
    AgentResult,
    AgentSource,
    AgentTask,
    AgentTaskKind,
    CreateAgentWorker,
    WorkerResponse,
)

WEB_RESEARCH_PROMPT = """
You are a task-oriented web research worker.

Scope:
- You work only with public web search and Wikipedia.
- Do not answer from memory when the tools do not support the claim.
- Respond in the same language as the task when possible.

Execution policy:
- For location, city, place, or organization questions, use wikipedia_search first to ground the subject.
- Then use duckduckgo_search once for broader evidence, if needed.
- Do not call the same tool repeatedly with near-identical queries unless the previous call returned zero useful results.
- After one or two useful tool calls, stop and return the final JSON answer.
- Never ask the user a follow-up question. Make reasonable assumptions and complete the assigned task.

Rules:
- Return status='success' only when the tools provide enough evidence.
- Return status='partial' when the available evidence is incomplete.
- Return status='failed' only when the task cannot be completed at all.
- Every successful or partial answer must include normalized sources.
- Keep the summary concise and focused on the assigned task.
- Return only valid JSON and no markdown fences.
""".strip()

_SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "by",
    "city",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "its",
    "of",
    "on",
    "or",
    "public",
    "sources",
    "strategic",
    "the",
    "to",
    "what",
    "with",
    "без",
    "в",
    "город",
    "для",
    "и",
    "из",
    "как",
    "какие",
    "какой",
    "кратко",
    "можно",
    "на",
    "о",
    "общественных",
    "общедоступные",
    "общедоступных",
    "опираясь",
    "по",
    "при",
    "развитие",
    "развития",
    "с",
    "стратегических",
    "стратегия",
    "учетом",
    "что",
    "это",
}


def _normalize_text(value: Any, *, max_chars: int = 320) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _query_keywords(query: str) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё-]{2,}", query.lower()):
        if token in _SEARCH_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def _rank_results(results: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    keywords = _query_keywords(query)
    if not keywords:
        return results

    scored: list[tuple[int, dict[str, Any]]] = []
    for item in results:
        haystack = " ".join(
            [
                str(item.get("title", "")),
                str(item.get("link", "")),
                str(item.get("snippet", "")),
            ]
        ).lower()
        score = sum(1 for keyword in keywords if keyword in haystack)
        if score > 0:
            scored.append((score, item))

    if not scored:
        return results

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored]


def build_duckduckgo_search_tool(
    *,
    max_results: int = 4,
    region: str = "wt-wt",
    timelimit: str | None = None,
) -> Any:
    api_wrapper = DuckDuckGoSearchAPIWrapper(
        region=region,
        time=timelimit,
        max_results=max_results,
        source="text",
    )

    @tool
    def duckduckgo_search(query: str) -> dict[str, list[dict[str, Any]]]:
        """Search the public web and return normalized search results."""
        results = api_wrapper.results(
            query=query,
            max_results=max_results,
            source="text",
        )
        results = _rank_results(results, query)
        normalized = [
            {
                "type": "web",
                "title": _normalize_text(item.get("title", ""), max_chars=120),
                "locator": item.get("link", ""),
                "snippet": _normalize_text(item.get("snippet", "")),
                "metadata": {
                    key: value
                    for key, value in item.items()
                    if key not in {"title", "link", "snippet"}
                },
            }
            for item in results[:max_results]
            if item.get("link")
        ]
        return {"results": normalized}

    return duckduckgo_search


def build_wikipedia_search_tool(
    *,
    lang: str = "en",
    top_k_results: int = 2,
    doc_content_chars_max: int = 1400,
) -> Any:
    api_wrapper = WikipediaAPIWrapper(
        lang=lang,
        top_k_results=top_k_results,
        doc_content_chars_max=doc_content_chars_max,
    )

    @tool
    def wikipedia_search(query: str) -> dict[str, list[dict[str, Any]]]:
        """Search Wikipedia and return normalized page summaries."""
        documents = api_wrapper.load(query)
        normalized = [
            {
                "type": "wikipedia",
                "title": document.metadata.get("title", query),
                "locator": document.metadata.get("source", ""),
                "snippet": _normalize_text(
                    document.metadata.get("summary", ""),
                    max_chars=500,
                ),
                "metadata": {
                    key: value
                    for key, value in document.metadata.items()
                    if key not in {"title", "source", "summary"}
                },
            }
            for document in documents
            if document.metadata.get("source")
        ]
        return {"results": normalized}

    return wikipedia_search


def build_web_research_tools(*, wiki_lang: str = "en") -> list[Any]:
    return [
        build_duckduckgo_search_tool(),
        build_wikipedia_search_tool(lang=wiki_lang),
    ]


class WebResearchAgent(CreateAgentWorker):
    name = "web_research_agent"
    capabilities: tuple[AgentTaskKind, ...] = ("web_research",)
    system_prompt = WEB_RESEARCH_PROMPT
    recursion_limit = 4

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        tools: Sequence[Any] | None = None,
        wiki_lang: str = "en",
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            tools=tools or build_web_research_tools(wiki_lang=wiki_lang),
            system_prompt=system_prompt,
        )

    def execute(self, task: AgentTask) -> AgentResult:
        if not self.supports(task):
            raise ValueError(
                f"{self.name} does not support task kind '{task.kind}'."
            )

        if self._can_use_manual_research():
            recovered = self._manual_research(task)
            if recovered is not None:
                return self._build_result(task, recovered)

        transcript: list[BaseMessage] = []
        error_message: str | None = None
        provisional_result: AgentResult | None = None
        try:
            for event in self._agent.stream(
                {"messages": [HumanMessage(self._format_task(task))]},
                config={
                    "configurable": {"thread_id": task.task_id},
                    "recursion_limit": self.recursion_limit,
                },
                stream_mode="updates",
            ):
                transcript.extend(self._messages_from_event(event))
            response = self._parse_worker_response({"messages": transcript})
            provisional_result = self._build_result(task, response)
        except GraphRecursionError as exc:
            error_message = str(exc).strip() or type(exc).__name__
        except Exception as exc:
            error_message = str(exc).strip() or type(exc).__name__
            if not transcript:
                return AgentResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    status="failed",
                    summary="",
                    sources=[],
                    artifacts={},
                    error=error_message,
                )

        if transcript and provisional_result is None:
            try:
                response = self._summarize_transcript(task, transcript)
                provisional_result = self._build_result(task, response)
            except Exception:
                pass

        if provisional_result and not self._should_recover(provisional_result):
            return provisional_result

        recovered = self._manual_research(task)
        if recovered is not None:
            return self._build_result(task, recovered)

        if provisional_result is not None:
            return provisional_result

        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status="failed",
            summary="",
            sources=[],
            artifacts={},
            error=error_message,
        )

    def _messages_from_event(self, event: dict[str, Any]) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        for update in event.values():
            if not isinstance(update, dict):
                continue
            update_messages = update.get("messages") or []
            messages.extend(
                message
                for message in update_messages
                if isinstance(message, BaseMessage)
            )
        return messages

    def _summarize_transcript(
        self,
        task: AgentTask,
        messages: Iterable[BaseMessage],
    ) -> WorkerResponse:
        sources = self._sources_from_messages(messages)
        if not sources:
            return WorkerResponse(status="failed", summary="", sources=[])
        return WorkerResponse(
            status=self._source_status(sources),
            summary=self._compose_summary(task, sources),
            sources=sources,
        )

    def _can_use_manual_research(self) -> bool:
        tool_names = {getattr(tool, "name", "") for tool in self.tools}
        return {"duckduckgo_search", "wikipedia_search"}.issubset(tool_names)

    def _should_recover(self, result: AgentResult) -> bool:
        if result.status != "success":
            return True
        return not any(source.type == "wikipedia" for source in result.sources)

    def _manual_research(self, task: AgentTask) -> WorkerResponse | None:
        plan = self._build_manual_search_plan(task)
        sources: list[AgentSource] = []

        wikipedia_query = str(plan.get("wikipedia_query", "")).strip()
        if wikipedia_query:
            payload = self._invoke_named_tool("wikipedia_search", wikipedia_query)
            sources.extend(self._sources_from_payload(payload, max_sources=2))

        for query in plan.get("web_queries", [])[:2]:
            payload = self._invoke_named_tool("duckduckgo_search", str(query))
            sources.extend(self._sources_from_payload(payload, max_sources=3))

        sources = self._dedupe_sources(sources, max_sources=5)
        if not sources:
            return None

        return WorkerResponse(
            status=self._source_status(sources),
            summary=self._compose_summary(task, sources),
            sources=sources,
        )

    def _build_manual_search_plan(self, task: AgentTask) -> dict[str, Any]:
        subject = self._guess_subject(task.query)
        query_lower = task.query.lower()
        if "санкт-петербург" in query_lower or "saint petersburg" in query_lower:
            web_queries = [f"{subject} Saint Petersburg transport"]
        else:
            web_queries = [f"{subject} transport"]
        if "туризм" in query_lower or "tourism" in query_lower:
            web_queries.append(f"{subject} tourism")
        return {
            "wikipedia_query": subject,
            "web_queries": list(dict.fromkeys(web_queries)),
        }

    def _guess_subject(self, query: str) -> str:
        focus_fragment = self._extract_focus_fragment(query)
        prompt = (
            "Return only the best English Wikipedia search spelling for the main focus subject.\n"
            "If the task asks how to develop one place with reference to another place, "
            "choose the place being developed, not the contextual reference.\n"
            "Do not add explanations.\n\n"
            f"Text:\n{focus_fragment or query}"
        )
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )
        if isinstance(content, str):
            subject = content.strip().strip("\"'")
            if subject:
                return subject
        return query.strip()

    def _extract_focus_fragment(self, query: str) -> str | None:
        patterns = [
            r"развивать\s+([A-Za-zА-Яа-яЁё-]+)",
            r"развитие\s+([A-Za-zА-Яа-яЁё-]+)",
            r"develop\s+([A-Za-z-]+)",
            r"development of\s+([A-Za-z-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _invoke_named_tool(self, tool_name: str, query: str) -> dict[str, Any]:
        tool = next(
            (candidate for candidate in self.tools if getattr(candidate, "name", None) == tool_name),
            None,
        )
        if tool is None or not query.strip():
            return {"results": []}
        try:
            return tool.invoke({"query": query})
        except Exception:
            return {"results": []}

    def _compose_summary(self, task: AgentTask, sources: list[AgentSource]) -> str:
        prompt = (
            "Answer the task using only the source snippets below.\n"
            "Respond in the same language as the task.\n"
            "Write a concise answer in 3-5 sentences.\n"
            "If the evidence is limited, say that the answer is preliminary.\n"
            "Do not output JSON, bullet points, or markdown.\n\n"
            f"Task:\n{task.query}\n\n"
            f"Sources:\n{json.dumps([source.model_dump(mode='json') for source in sources], ensure_ascii=False, indent=2)}"
        )
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )
        if isinstance(content, str) and content.strip():
            return _normalize_text(content, max_chars=900)
        return (
            "Собраны внешние источники по запросу. Ответ носит предварительный "
            "характер и опирается на найденные материалы."
        )

    def _source_status(self, sources: list[AgentSource]) -> str:
        has_wikipedia = any(source.type == "wikipedia" for source in sources)
        if has_wikipedia and len(sources) >= 2:
            return "success"
        if sources:
            return "partial"
        return "failed"

    def _sources_from_payload(
        self,
        payload: dict[str, Any],
        *,
        max_sources: int,
    ) -> list[AgentSource]:
        sources: list[AgentSource] = []
        for item in payload.get("results", []):
            try:
                source = AgentSource.model_validate(
                    {
                        "type": item.get("type", "web"),
                        "title": _normalize_text(item.get("title", ""), max_chars=120),
                        "locator": item.get("locator", ""),
                        "snippet": _normalize_text(item.get("snippet", "")),
                        "metadata": item.get("metadata", {}),
                    }
                )
            except Exception:
                continue
            if source.locator:
                sources.append(source)
            if len(sources) >= max_sources:
                break
        return sources

    def _dedupe_sources(
        self,
        sources: list[AgentSource],
        *,
        max_sources: int,
    ) -> list[AgentSource]:
        deduped: list[AgentSource] = []
        seen: set[tuple[str, str]] = set()
        for source in sources:
            key = (source.type, source.locator)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
            if len(deduped) >= max_sources:
                break
        return deduped

    def _sources_from_messages(
        self,
        messages: Iterable[BaseMessage],
        *,
        max_sources: int = 5,
    ) -> list[AgentSource]:
        sources: list[AgentSource] = []
        seen: set[tuple[str, str]] = set()

        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            try:
                payload = json.loads(message.content)
            except Exception:
                continue

            for item in payload.get("results", []):
                try:
                    source = AgentSource.model_validate(
                        {
                            "type": item.get("type", message.name or "web"),
                            "title": _normalize_text(item.get("title", ""), max_chars=120),
                            "locator": item.get("locator", ""),
                            "snippet": _normalize_text(item.get("snippet", "")),
                            "metadata": item.get("metadata", {}),
                        }
                    )
                except Exception:
                    continue

                key = (source.type, source.locator)
                if not source.locator or key in seen:
                    continue
                seen.add(key)
                sources.append(source)
                if len(sources) >= max_sources:
                    return sources

        return sources


def create_web_research_agent(**kwargs: Any) -> WebResearchAgent:
    return WebResearchAgent(**kwargs)


__all__ = [
    "WebResearchAgent",
    "build_duckduckgo_search_tool",
    "build_web_research_tools",
    "build_wikipedia_search_tool",
    "create_web_research_agent",
]
