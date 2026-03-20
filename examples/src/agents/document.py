from __future__ import annotations

from typing import Any, Callable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel

from .runtime import AgentTaskKind, CreateAgentWorker

DOCUMENT_RESEARCH_PROMPT = """
You are a task-oriented document research worker.

Scope:
- You work only with the provided document search tool.
- Do not use outside knowledge and do not invent missing facts.
- Respond in the same language as the task when possible.

Rules:
- Base the answer only on retrieved document fragments.
- Return status='success' only when the retrieved fragments are sufficient.
- Return status='partial' when the retrieved evidence is weak or incomplete.
- Return status='failed' only when the task cannot be completed at all.
- Include normalized sources for every supported answer.
- Keep the summary concise and directly focused on the assigned task.
""".strip()


def resolve_document_research_tool(
    *,
    store: Any | None = None,
    tool: Any | None = None,
    tool_factory: Callable[[Any], Any] | None = None,
) -> Any:
    if tool is not None:
        return tool
    if store is None:
        raise ValueError("Either 'store' or 'tool' must be provided.")
    if tool_factory is not None:
        return tool_factory(store)
    if hasattr(store, "research_tool"):
        return store.research_tool
    if hasattr(store, "tool"):
        return store.tool
    raise ValueError("Could not resolve a document research tool from the store.")


class DocumentResearchAgent(CreateAgentWorker):
    name = "document_research_agent"
    capabilities: tuple[AgentTaskKind, ...] = ("document_research",)
    system_prompt = DOCUMENT_RESEARCH_PROMPT

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        store: Any | None = None,
        tool: Any | None = None,
        tool_factory: Callable[[Any], Any] | None = None,
        tools: Sequence[Any] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        resolved_tools = list(tools) if tools is not None else [
            resolve_document_research_tool(
                store=store,
                tool=tool,
                tool_factory=tool_factory,
            )
        ]
        super().__init__(
            model=model,
            tools=resolved_tools,
            system_prompt=system_prompt,
        )


def create_document_research_agent(**kwargs: Any) -> DocumentResearchAgent:
    return DocumentResearchAgent(**kwargs)


__all__ = [
    "DocumentResearchAgent",
    "create_document_research_agent",
    "resolve_document_research_tool",
]
