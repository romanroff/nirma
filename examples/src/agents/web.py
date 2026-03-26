from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html import unescape
from typing import Any, Callable, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from langchain.tools import tool
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel

from .. import trace_log
from .runtime import (
    AgentResult,
    AgentSource,
    AgentTask,
    AgentTaskKind,
    CreateAgentWorker,
    WorkerResponse,
)

try:  # pragma: no cover - optional dependency in tests
    import trafilatura
except Exception:  # pragma: no cover - defensive import guard
    trafilatura = None

WEB_RESEARCH_PROMPT = """
You are a deterministic web research worker.

Scope:
- You work with public web search, HTTP page fetches, optional Wikipedia grounding,
  and page text extraction.
- Do not answer from memory when the evidence does not support the claim.
- Respond in the same language as the task when possible.

Rules:
- Plan a few diverse search queries, then synthesize only from extracted evidence.
- Prefer page content over search snippets. Use snippets only as a last-resort fallback.
- Keep the final answer concise and focused on the assigned task.
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
    "что",
    "это",
}
_NON_HTML_EXTENSIONS = {
    ".7z",
    ".avi",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".json",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".svg",
    ".tsv",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}
_TRACKING_QUERY_KEYS = {"fbclid", "gclid", "ref", "ref_src", "source"}
_USER_AGENT = (
    "Mozilla/5.0 (compatible; nirma-web-research/1.0; "
    "+https://example.invalid/nirma)"
)


@dataclass(slots=True)
class SearchCandidate:
    type: str
    title: str
    locator: str
    normalized_locator: str
    snippet: str
    metadata: dict[str, Any]
    search_query: str
    rank: int
    domain: str
    score: float = 0.0


@dataclass(slots=True)
class FetchResult:
    requested_url: str
    final_url: str
    status_code: int | None
    content_type: str
    html: str
    error: str | None = None


@dataclass(slots=True)
class Passage:
    type: str
    title: str
    locator: str
    text: str
    metadata: dict[str, Any]
    score: float
    origin: str = "page"


def _normalize_text(value: Any, *, max_chars: int = 320) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _query_keywords(query: str) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9-]{2,}", query.lower()):
        if token in _SEARCH_STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def _message_text(content: Any) -> str:
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )
    if isinstance(content, str):
        return content.strip()
    return ""


def _normalize_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except Exception:
        return ""

    scheme = (parts.scheme or "https").lower()
    if scheme not in {"http", "https"}:
        return ""

    netloc = parts.netloc.lower()
    if not netloc:
        return ""

    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    kept_query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered.startswith("utm_") or lowered in _TRACKING_QUERY_KEYS:
            continue
        kept_query.append((key, value))

    query = urlencode(kept_query, doseq=True)
    return urlunsplit((scheme, netloc, path or "/", query, ""))


def _domain_from_url(url: str) -> str:
    try:
        return urlsplit(url).netloc.lower()
    except Exception:
        return ""


def _path_from_url(url: str) -> str:
    try:
        return urlsplit(url).path.lower() or "/"
    except Exception:
        return "/"


def _has_non_html_extension(url: str) -> bool:
    path = _path_from_url(url)
    return any(path.endswith(extension) for extension in _NON_HTML_EXTENSIONS)


def _looks_like_low_value_path(url: str) -> bool:
    path = _path_from_url(url)
    if path in {"", "/"}:
        return True
    low_value_tokens = (
        "/archive",
        "/author/",
        "/authors/",
        "/blog",
        "/category/",
        "/categories/",
        "/feed",
        "/listing",
        "/listings",
        "/page/",
        "/search",
        "/tag/",
        "/tags/",
        "/topic/",
        "/topics/",
    )
    return any(token in path for token in low_value_tokens)


def _score_keyword_hits(text: str, keywords: list[str]) -> float:
    haystack = text.lower()
    return float(sum(1 for keyword in keywords if keyword in haystack))


def _chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    cleaned = _normalize_text(text, max_chars=max(len(str(text or "")), chunk_size))
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _extract_json_object(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    raise ValueError("Could not locate a JSON object in the model response.")


def _simple_extract_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript|svg).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_duckduckgo_search_tool(
    *,
    max_results: int = 8,
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
    top_k_results: int = 3,
    doc_content_chars_max: int = 1800,
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


def build_web_research_tools(
    *,
    wiki_lang: str = "en",
    search_results_per_query: int = 8,
) -> list[Any]:
    return [
        build_duckduckgo_search_tool(max_results=search_results_per_query),
        build_wikipedia_search_tool(lang=wiki_lang),
    ]


class WebResearchAgent(CreateAgentWorker):
    name = "web_research_agent"
    capabilities: tuple[AgentTaskKind, ...] = ("web_research",)
    system_prompt = WEB_RESEARCH_PROMPT
    max_response_size_bytes = 2 * 1024 * 1024
    search_backend: Callable[..., dict[str, list[dict[str, Any]]]] | None
    page_fetcher: Callable[..., dict[str, Any]] | None
    content_extractor: Callable[[str, str], str] | None

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        tools: Sequence[Any] | None = None,
        wiki_lang: str = "en",
        system_prompt: str | None = None,
        search_query_count: int = 3,
        search_results_per_query: int = 8,
        fetch_budget: int = 10,
        per_domain_cap: int = 2,
        request_timeout_sec: int = 10,
        max_final_sources: int = 8,
        wiki_enabled: bool = True,
        search_backend: Callable[..., dict[str, list[dict[str, Any]]]] | None = None,
        page_fetcher: Callable[..., dict[str, Any]] | None = None,
        content_extractor: Callable[[str, str], str] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            tools=tools or build_web_research_tools(
                wiki_lang=wiki_lang,
                search_results_per_query=search_results_per_query,
            ),
            system_prompt=system_prompt,
        )
        self.search_query_count = max(1, int(search_query_count))
        self.search_results_per_query = max(1, int(search_results_per_query))
        self.fetch_budget = max(1, int(fetch_budget))
        self.per_domain_cap = max(1, int(per_domain_cap))
        self.request_timeout_sec = max(1, int(request_timeout_sec))
        self.max_final_sources = max(1, int(max_final_sources))
        self.wiki_enabled = wiki_enabled
        self.search_backend = search_backend
        self.page_fetcher = page_fetcher
        self.content_extractor = content_extractor

    def execute(self, task: AgentTask) -> AgentResult:
        if not self.supports(task):
            raise ValueError(
                f"{self.name} does not support task kind '{task.kind}'."
            )

        trace_log.log_event(
            "task_worker_start",
            worker_name=self.name,
            task=task,
            tools=[getattr(tool, "name", repr(tool)) for tool in self.tools],
        )
        artifacts: dict[str, Any] = {
            "search_queries": [],
            "candidate_urls": [],
            "fetched_urls": [],
            "skipped_urls": [],
            "used_passages": [],
            "counts": {},
        }
        try:
            search_queries = self._plan_search_queries(task)
            wiki_query = self._build_wikipedia_query(task)
            artifacts["search_queries"] = search_queries
            trace_log.log_event(
                "web_research_search_plan",
                task_id=task.task_id,
                search_queries=search_queries,
                wikipedia_query=wiki_query,
            )

            candidates = self._collect_candidates(
                search_queries=search_queries,
                wiki_query=wiki_query,
            )
            ranked_candidates = self._rank_candidates(task, candidates)
            selected_candidates = ranked_candidates[: self.fetch_budget]
            artifacts["candidate_urls"] = [
                self._candidate_artifact(candidate)
                for candidate in selected_candidates
            ]
            trace_log.log_event(
                "web_research_candidate_urls",
                task_id=task.task_id,
                candidate_urls=artifacts["candidate_urls"],
            )

            passages, fallback_passages, fetch_artifacts, skipped_urls = (
                self._fetch_and_extract(task, selected_candidates, search_queries)
            )
            artifacts["fetched_urls"] = fetch_artifacts
            artifacts["skipped_urls"] = skipped_urls

            used_passages = self._select_used_passages(passages, fallback_passages)
            artifacts["used_passages"] = [
                {
                    "locator": passage.locator,
                    "title": passage.title,
                    "score": round(passage.score, 3),
                    "origin": passage.origin,
                    "excerpt": _normalize_text(passage.text, max_chars=220),
                }
                for passage in used_passages
            ]

            sources = self._build_sources(used_passages)
            summary = self._synthesize_summary(task, used_passages)
            status = self._resolve_status(used_passages, sources)
            response = WorkerResponse(
                status=status,
                summary=summary,
                sources=sources,
            )

            page_locators = {
                passage.locator
                for passage in used_passages
                if passage.origin == "page"
            }
            artifacts["counts"] = {
                "candidate_count": len(candidates),
                "selected_candidate_count": len(selected_candidates),
                "page_passage_count": len(
                    [passage for passage in passages if passage.origin == "page"]
                ),
                "fallback_passage_count": len(fallback_passages),
                "used_passage_count": len(used_passages),
                "page_count": len(page_locators),
                "source_count": len(sources),
            }

            result = super()._build_result(task, response).model_copy(
                update={"artifacts": artifacts}
            )
            trace_log.log_event(
                "task_worker_result",
                worker_name=self.name,
                task_id=task.task_id,
                result=result,
            )
            return result
        except Exception as exc:
            error_message = str(exc).strip() or type(exc).__name__
            trace_log.log_event(
                "task_worker_error",
                worker_name=self.name,
                task_id=task.task_id,
                task=task,
                error=exc,
                artifacts=artifacts,
            )
            result = AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status="failed",
                summary="",
                sources=[],
                artifacts=artifacts,
                error=error_message,
            )
            trace_log.log_event(
                "task_worker_result",
                worker_name=self.name,
                task_id=task.task_id,
                result=result,
            )
            return result

    def _plan_search_queries(self, task: AgentTask) -> list[str]:
        prompt = (
            "Return only valid JSON with the shape "
            "{\"queries\": [\"query 1\", \"query 2\", \"query 3\"]}.\n"
            "Generate concise, diverse public-web search queries for research.\n"
            "Use the same language as the task when practical.\n"
            "Avoid near-duplicates. Prefer queries likely to surface factual sources.\n\n"
            f"Task:\n{task.query}"
        )
        fallback_queries = self._fallback_search_queries(task.query)
        try:
            raw_text = self._invoke_model_text(prompt)
            payload = json.loads(_extract_json_object(raw_text))
            queries = [
                _normalize_text(query, max_chars=140)
                for query in payload.get("queries", [])
                if str(query).strip()
            ]
        except Exception:
            queries = []

        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries + fallback_queries:
            normalized = str(query).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
            if len(deduped) >= self.search_query_count:
                break
        return deduped or [task.query.strip()]

    def _fallback_search_queries(self, query: str) -> list[str]:
        subject = self._extract_focus_fragment(query) or _normalize_text(
            query,
            max_chars=100,
        )
        variants = [
            query.strip(),
            f"{subject} official statistics".strip(),
            f"{subject} analysis report".strip(),
        ]
        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            if not variant:
                continue
            lowered = variant.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(variant)
        return deduped

    def _build_wikipedia_query(self, task: AgentTask) -> str:
        if not self.wiki_enabled:
            return ""
        focus = self._extract_focus_fragment(task.query)
        if focus:
            return focus
        keywords = _query_keywords(task.query)
        if keywords:
            return " ".join(keywords[:5])
        return task.query.strip()

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

    def _collect_candidates(
        self,
        *,
        search_queries: list[str],
        wiki_query: str,
    ) -> list[SearchCandidate]:
        candidates: list[SearchCandidate] = []
        for query in search_queries[: self.search_query_count]:
            payload = self._search(
                query,
                source="duckduckgo",
                max_results=self.search_results_per_query,
            )
            candidates.extend(
                self._candidates_from_payload(
                    payload,
                    search_query=query,
                )
            )

        if self.wiki_enabled and wiki_query:
            payload = self._search(
                wiki_query,
                source="wikipedia",
                max_results=3,
            )
            candidates.extend(
                self._candidates_from_payload(
                    payload,
                    search_query=wiki_query,
                )
            )
        return candidates

    def _search(
        self,
        query: str,
        *,
        source: str,
        max_results: int,
    ) -> dict[str, list[dict[str, Any]]]:
        if self.search_backend is not None:
            return self.search_backend(
                query,
                source=source,
                max_results=max_results,
            )
        tool_name = {
            "duckduckgo": "duckduckgo_search",
            "wikipedia": "wikipedia_search",
        }.get(source)
        if tool_name is None:
            return {"results": []}
        payload = self._invoke_named_tool(tool_name, query)
        results = list(payload.get("results", []))
        return {"results": results[:max_results]}

    def _invoke_named_tool(self, tool_name: str, query: str) -> dict[str, Any]:
        tool = next(
            (
                candidate
                for candidate in self.tools
                if getattr(candidate, "name", None) == tool_name
            ),
            None,
        )
        if tool is None or not query.strip():
            return {"results": []}
        try:
            return tool.invoke({"query": query})
        except Exception:
            return {"results": []}

    def _candidates_from_payload(
        self,
        payload: dict[str, Any],
        *,
        search_query: str,
    ) -> list[SearchCandidate]:
        candidates: list[SearchCandidate] = []
        for rank, item in enumerate(payload.get("results", []), start=1):
            locator = _normalize_url(item.get("locator", ""))
            if not locator:
                continue
            candidates.append(
                SearchCandidate(
                    type=str(item.get("type", "web")),
                    title=_normalize_text(item.get("title", ""), max_chars=180),
                    locator=locator,
                    normalized_locator=locator,
                    snippet=_normalize_text(item.get("snippet", ""), max_chars=500),
                    metadata=dict(item.get("metadata", {})),
                    search_query=search_query,
                    rank=rank,
                    domain=_domain_from_url(locator),
                )
            )
        return candidates

    def _rank_candidates(
        self,
        task: AgentTask,
        candidates: list[SearchCandidate],
    ) -> list[SearchCandidate]:
        if not candidates:
            return []

        keywords = _query_keywords(
            " ".join([task.query, *(candidate.search_query for candidate in candidates)])
        )
        deduped: dict[str, SearchCandidate] = {}
        for candidate in candidates:
            haystack = " ".join(
                [
                    candidate.title,
                    candidate.snippet,
                    candidate.normalized_locator,
                    candidate.search_query,
                ]
            )
            score = _score_keyword_hits(haystack, keywords)
            score += max(0, self.search_results_per_query - candidate.rank) * 0.35
            score += 1.0 if candidate.type == "wikipedia" else 0.0
            score += 0.5 if candidate.snippet else 0.0
            if _looks_like_low_value_path(candidate.normalized_locator):
                score -= 2.0
            if _has_non_html_extension(candidate.normalized_locator):
                score -= 10.0
            candidate.score = score

            previous = deduped.get(candidate.normalized_locator)
            if previous is None or candidate.score > previous.score:
                deduped[candidate.normalized_locator] = candidate

        ordered = sorted(
            deduped.values(),
            key=lambda candidate: (
                -candidate.score,
                candidate.rank,
                candidate.normalized_locator,
            ),
        )

        capped: list[SearchCandidate] = []
        domain_counts: dict[str, int] = {}
        for candidate in ordered:
            domain = candidate.domain or "_unknown"
            count = domain_counts.get(domain, 0)
            if count >= self.per_domain_cap:
                continue
            domain_counts[domain] = count + 1
            capped.append(candidate)
        return capped

    def _candidate_artifact(self, candidate: SearchCandidate) -> dict[str, Any]:
        return {
            "type": candidate.type,
            "title": candidate.title,
            "locator": candidate.locator,
            "score": round(candidate.score, 3),
            "domain": candidate.domain,
            "search_query": candidate.search_query,
        }

    def _fetch_and_extract(
        self,
        task: AgentTask,
        candidates: list[SearchCandidate],
        search_queries: list[str],
    ) -> tuple[list[Passage], list[Passage], list[dict[str, Any]], list[dict[str, Any]]]:
        fetchable: list[SearchCandidate] = []
        skipped_urls: list[dict[str, Any]] = []
        for candidate in candidates:
            if _has_non_html_extension(candidate.locator):
                skipped_urls.append(
                    {
                        "locator": candidate.locator,
                        "reason": "non_html_extension",
                    }
                )
                continue
            fetchable.append(candidate)

        if not fetchable:
            return [], [], [], skipped_urls

        max_workers = min(5, max(1, len(fetchable)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fetch_results = list(executor.map(self._fetch_candidate, fetchable))

        keywords = _query_keywords(" ".join([task.query, *search_queries]))
        passages: list[Passage] = []
        fallback_passages: list[Passage] = []
        fetched_urls: list[dict[str, Any]] = []

        for candidate, fetch_result in zip(fetchable, fetch_results):
            fetched_urls.append(
                {
                    "requested_url": fetch_result.requested_url,
                    "final_url": fetch_result.final_url,
                    "status_code": fetch_result.status_code,
                    "content_type": fetch_result.content_type,
                    "error": fetch_result.error,
                }
            )
            if fetch_result.error:
                skipped_urls.append(
                    {
                        "locator": candidate.locator,
                        "reason": fetch_result.error,
                    }
                )
                if candidate.snippet:
                    fallback_passages.append(
                        self._snippet_fallback_passage(candidate, keywords)
                    )
                continue

            extracted = self._extract_content(fetch_result.final_url, fetch_result.html)
            if not extracted:
                skipped_urls.append(
                    {
                        "locator": candidate.locator,
                        "reason": "empty_extraction",
                    }
                )
                if candidate.snippet:
                    fallback_passages.append(
                        self._snippet_fallback_passage(candidate, keywords)
                    )
                continue

            passages.extend(
                self._select_page_passages(
                    candidate=candidate,
                    locator=fetch_result.final_url,
                    text=extracted,
                    keywords=keywords,
                )
            )

        trace_log.log_event(
            "web_research_fetch_summary",
            fetched_urls=fetched_urls,
            skipped_urls=skipped_urls,
        )
        return passages, fallback_passages, fetched_urls, skipped_urls

    def _fetch_candidate(self, candidate: SearchCandidate) -> FetchResult:
        payload = (
            self.page_fetcher(candidate.locator, timeout=self.request_timeout_sec)
            if self.page_fetcher is not None
            else self._default_page_fetcher(candidate.locator, timeout=self.request_timeout_sec)
        )
        return FetchResult(
            requested_url=str(payload.get("requested_url", candidate.locator)),
            final_url=_normalize_url(
                str(payload.get("final_url", payload.get("url", candidate.locator)))
            )
            or candidate.locator,
            status_code=payload.get("status_code"),
            content_type=str(payload.get("content_type", "")),
            html=str(payload.get("html", "")),
            error=payload.get("error"),
        )

    def _default_page_fetcher(self, url: str, *, timeout: int) -> dict[str, Any]:
        try:
            with requests.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=timeout,
                allow_redirects=True,
                stream=True,
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                if response.status_code >= 400:
                    return {
                        "requested_url": url,
                        "final_url": response.url or url,
                        "status_code": response.status_code,
                        "content_type": content_type,
                        "html": "",
                        "error": f"http_{response.status_code}",
                    }
                if "html" not in content_type.lower():
                    return {
                        "requested_url": url,
                        "final_url": response.url or url,
                        "status_code": response.status_code,
                        "content_type": content_type,
                        "html": "",
                        "error": "non_html_content_type",
                    }

                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > self.max_response_size_bytes:
                        return {
                            "requested_url": url,
                            "final_url": response.url or url,
                            "status_code": response.status_code,
                            "content_type": content_type,
                            "html": "",
                            "error": "response_too_large",
                        }
                    chunks.append(chunk)

                body = b"".join(chunks).decode(
                    response.encoding or "utf-8",
                    errors="ignore",
                )
                return {
                    "requested_url": url,
                    "final_url": response.url or url,
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "html": body,
                    "error": None,
                }
        except requests.Timeout:
            return {
                "requested_url": url,
                "final_url": url,
                "status_code": None,
                "content_type": "",
                "html": "",
                "error": "timeout",
            }
        except requests.RequestException as exc:
            return {
                "requested_url": url,
                "final_url": url,
                "status_code": None,
                "content_type": "",
                "html": "",
                "error": exc.__class__.__name__.lower(),
            }

    def _extract_content(self, url: str, html: str) -> str:
        if self.content_extractor is not None:
            extracted = self.content_extractor(url, html)
            return _normalize_text(
                extracted,
                max_chars=max(len(str(extracted or "")), 1200),
            )

        extracted = ""
        if trafilatura is not None:
            try:  # pragma: no cover - depends on optional library internals
                extracted = trafilatura.extract(
                    html,
                    url=url,
                    include_comments=False,
                    include_links=False,
                    include_tables=False,
                    favor_recall=True,
                ) or ""
            except Exception:
                extracted = ""
        if not extracted:
            extracted = _simple_extract_text(html)
        return _normalize_text(extracted, max_chars=max(len(extracted), 4000))

    def _select_page_passages(
        self,
        *,
        candidate: SearchCandidate,
        locator: str,
        text: str,
        keywords: list[str],
    ) -> list[Passage]:
        chunks = _chunk_text(text, chunk_size=1200, overlap=150)
        if not chunks:
            return []

        scored_chunks: list[tuple[float, str]] = []
        for index, chunk in enumerate(chunks):
            score = _score_keyword_hits(chunk, keywords)
            score += _score_keyword_hits(candidate.title, keywords) * 0.2
            if index == 0:
                score += 0.15
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        selected: list[Passage] = []
        for score, chunk in scored_chunks[:2]:
            selected.append(
                Passage(
                    type=candidate.type,
                    title=candidate.title,
                    locator=locator,
                    text=chunk,
                    metadata={
                        **candidate.metadata,
                        "search_query": candidate.search_query,
                    },
                    score=score,
                    origin="page",
                )
            )
        return selected

    def _snippet_fallback_passage(
        self,
        candidate: SearchCandidate,
        keywords: list[str],
    ) -> Passage:
        return Passage(
            type=candidate.type,
            title=candidate.title,
            locator=candidate.locator,
            text=candidate.snippet,
            metadata={
                **candidate.metadata,
                "search_query": candidate.search_query,
            },
            score=_score_keyword_hits(candidate.snippet, keywords) - 1.0,
            origin="snippet",
        )

    def _select_used_passages(
        self,
        passages: list[Passage],
        fallback_passages: list[Passage],
    ) -> list[Passage]:
        ordered_pages = sorted(passages, key=lambda passage: passage.score, reverse=True)
        selected = ordered_pages[:12]
        if len(selected) >= 12:
            trace_log.log_event(
                "web_research_used_passages",
                used_passages=[passage.locator for passage in selected],
            )
            return selected

        ordered_fallbacks = sorted(
            fallback_passages,
            key=lambda passage: passage.score,
            reverse=True,
        )
        seen = {passage.locator for passage in selected}
        for passage in ordered_fallbacks:
            if passage.locator in seen:
                continue
            seen.add(passage.locator)
            selected.append(passage)
            if len(selected) >= 12:
                break

        trace_log.log_event(
            "web_research_used_passages",
            used_passages=[
                {
                    "locator": passage.locator,
                    "origin": passage.origin,
                }
                for passage in selected
            ],
        )
        return selected

    def _build_sources(self, passages: list[Passage]) -> list[AgentSource]:
        sources: list[AgentSource] = []
        seen: set[str] = set()
        ordered = sorted(
            passages,
            key=lambda passage: (
                passage.origin != "page",
                -passage.score,
                passage.locator,
            ),
        )
        for passage in ordered:
            if passage.locator in seen:
                continue
            seen.add(passage.locator)
            sources.append(
                AgentSource(
                    type=passage.type,
                    title=_normalize_text(passage.title, max_chars=120),
                    locator=passage.locator,
                    snippet=_normalize_text(passage.text, max_chars=280),
                    metadata=dict(passage.metadata),
                )
            )
            if len(sources) >= self.max_final_sources:
                break
        return sources

    def _synthesize_summary(self, task: AgentTask, passages: list[Passage]) -> str:
        if not passages:
            return ""

        prompt = (
            "Answer the task using only the evidence passages below.\n"
            "Respond in the same language as the task.\n"
            "Write a concise answer in 3-5 sentences.\n"
            "If the evidence is thin or mixed, clearly say the answer is preliminary.\n"
            "Do not use markdown, bullets, or JSON.\n\n"
            f"Task:\n{task.query}\n\n"
            f"Evidence:\n{json.dumps([self._passage_payload(passage) for passage in passages], ensure_ascii=False, indent=2)}"
        )
        try:
            text = self._invoke_model_text(prompt)
        except Exception:
            text = ""
        if text:
            return _normalize_text(text, max_chars=900)
        return _normalize_text(
            "Собраны внешние материалы по запросу. Ответ предварительный и основан "
            "на извлеченных фрагментах страниц.",
            max_chars=900,
        )

    def _invoke_model_text(self, prompt: str) -> str:
        response = self._model.invoke(prompt)
        return _message_text(response.content)

    def _passage_payload(self, passage: Passage) -> dict[str, Any]:
        return {
            "type": passage.type,
            "title": passage.title,
            "locator": passage.locator,
            "origin": passage.origin,
            "score": round(passage.score, 3),
            "text": _normalize_text(passage.text, max_chars=500),
        }

    def _resolve_status(
        self,
        passages: list[Passage],
        sources: list[AgentSource],
    ) -> str:
        page_locators = {
            passage.locator
            for passage in passages
            if passage.origin == "page"
        }
        if len(page_locators) >= 6 and len(sources) >= 5:
            return "success"
        if passages:
            return "partial"
        return "failed"


def create_web_research_agent(**kwargs: Any) -> WebResearchAgent:
    return WebResearchAgent(**kwargs)


__all__ = [
    "WebResearchAgent",
    "build_duckduckgo_search_tool",
    "build_web_research_tools",
    "build_wikipedia_search_tool",
    "create_web_research_agent",
]
