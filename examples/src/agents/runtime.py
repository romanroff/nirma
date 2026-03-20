from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, Sequence
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from pydantic import Field

from ..llms import agent_llm
from ..model import Model

AgentTaskKind = Literal["web_research", "document_research"]
AgentStatus = Literal["success", "partial", "failed"]


class AgentSource(Model):
    type: str = Field(description="Source type.")
    title: str = Field(description="Short source title.")
    locator: str = Field(description="Resolvable source locator such as URL or page ref.")
    snippet: str = Field(description="Short supporting snippet from the source.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional source metadata.",
    )


class AgentTask(Model):
    task_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Stable task identifier.",
    )
    kind: AgentTaskKind = Field(description="Task kind.")
    query: str = Field(description="Primary task query.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional task context.",
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional task constraints.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form task metadata.",
    )


class AgentResult(Model):
    task_id: str = Field(description="Task identifier.")
    agent_name: str = Field(description="Worker name.")
    status: AgentStatus = Field(description="Execution status.")
    summary: str = Field(description="Short task result summary.")
    sources: list[AgentSource] = Field(
        default_factory=list,
        description="Normalized supporting sources.",
    )
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional execution artifacts.",
    )
    error: str | None = Field(
        default=None,
        description="Execution error if one happened.",
    )


class WorkerResponse(Model):
    status: AgentStatus = Field(description="Worker completion status.")
    summary: str = Field(description="Short answer focused on the task.")
    sources: list[AgentSource] = Field(
        default_factory=list,
        description="Supporting sources used by the worker.",
    )


class TaskAgentRuntime(ABC):
    name: ClassVar[str]
    capabilities: ClassVar[tuple[AgentTaskKind, ...]]

    @abstractmethod
    def supports(self, task: AgentTask) -> bool:
        raise NotImplementedError

    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResult:
        raise NotImplementedError


class CreateAgentWorker(TaskAgentRuntime):
    name: ClassVar[str]
    capabilities: ClassVar[tuple[AgentTaskKind, ...]]
    system_prompt: ClassVar[str]

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        tools: Sequence[Any],
        system_prompt: str | None = None,
    ) -> None:
        if not tools:
            raise ValueError("At least one tool must be provided.")

        self._model = model or agent_llm
        self._tools = tuple(tools)
        self._system_prompt = system_prompt or self.system_prompt
        self._agent = create_agent(
            model=self._model,
            tools=list(self._tools),
            system_prompt=self._system_prompt,
            name=self.name,
        )

    @property
    def tools(self) -> tuple[Any, ...]:
        return self._tools

    def supports(self, task: AgentTask) -> bool:
        return task.kind in self.capabilities

    def execute(self, task: AgentTask) -> AgentResult:
        if not self.supports(task):
            raise ValueError(
                f"{self.name} does not support task kind '{task.kind}'."
            )

        try:
            raw_result = self._agent.invoke(
                input={"messages": [HumanMessage(self._format_task(task))]},
                config={"configurable": {"thread_id": task.task_id}},
            )
            structured = raw_result.get("structured_response")
            if structured is None:
                response = self._parse_worker_response(raw_result)
                return self._build_result(task, response)

            response = WorkerResponse.model_validate(structured)
            return self._build_result(task, response)
        except Exception as exc:
            error_message = str(exc).strip() or type(exc).__name__
            return AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status="failed",
                summary="",
                sources=[],
                artifacts={},
                error=error_message,
            )

    def _build_result(self, task: AgentTask, response: WorkerResponse) -> AgentResult:
        status = response.status
        if status == "success" and not response.sources:
            status = "partial"

        error = None
        if status == "failed":
            error = "Agent reported that it could not complete the task."

        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status=status,
            summary=response.summary,
            sources=response.sources,
            artifacts={},
            error=error,
        )

    def _parse_worker_response(self, raw_result: dict[str, Any]) -> WorkerResponse:
        messages = raw_result.get("messages") or []
        if not messages:
            raise ValueError("Agent returned no messages.")

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            raise ValueError("Agent did not finish with an AI message.")

        content = last_message.content
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Agent returned an empty final message.")

        parsed = self._extract_json_object(content)
        return WorkerResponse.model_validate_json(parsed)

    def _extract_json_object(self, content: str) -> str:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return cleaned

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1]
        raise ValueError(f"Could not parse JSON response from agent: {content[:200]}")

    def _format_task(self, task: AgentTask) -> str:
        payload = {
            "task_id": task.task_id,
            "kind": task.kind,
            "query": task.query,
            "context": task.context,
            "constraints": task.constraints,
            "metadata": task.metadata,
        }
        return (
            "Complete the task using only your available tools. "
            "Return only valid JSON with this shape and no markdown fences:\n"
            "{"
            "\"status\":\"success|partial|failed\","
            "\"summary\":\"short task-focused answer\","
            "\"sources\":[{"
            "\"type\":\"source type\","
            "\"title\":\"short title\","
            "\"locator\":\"url or locator\","
            "\"snippet\":\"short supporting snippet\","
            "\"metadata\":{}"
            "}]"
            "}\n\n"
            f"Task:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )


__all__ = [
    "AgentResult",
    "AgentSource",
    "AgentStatus",
    "AgentTask",
    "AgentTaskKind",
    "CreateAgentWorker",
    "TaskAgentRuntime",
    "WorkerResponse",
]
