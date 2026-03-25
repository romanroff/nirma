from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...board import BaseNote, Board
from ...store import Store
from ... import trace_log
from ...utils import get_id, iter_document_paths
from ..document import create_document_research_agent
from ..role import Role
from ..runtime import AgentResult, AgentTask, TaskAgentRuntime
from ..web import create_web_research_agent

_SUMMARY_LIMIT = 280
_CONTENT_LIMIT = 2000
_KEYWORD_LIMIT = 5
_SOURCE_LIMIT = 3


class TaskResearchExpertAdapter:
    response_format = BaseNote

    def __init__(
        self,
        *,
        board: Board,
        worker: TaskAgentRuntime,
        role: Role,
        task_kind: str,
        document_path: str | Path | None = None,
    ) -> None:
        self.board = board
        self.worker = worker
        self.role = role
        self.task_kind = task_kind
        self.document_path = Path(document_path) if document_path is not None else None
        self.id = get_id()

    @property
    def info(self) -> dict[str, str]:
        return {
            "id": self.id,
            "role_name": self.role.name,
            "role_description": self.role.description,
        }

    def invoke(self, messages=None, thread_id: str | None = None, force: bool = False) -> BaseNote:
        del messages, thread_id, force

        task = self._build_task()
        trace_log.log_event(
            "research_expert_start",
            expert_id=self.id,
            role=self.role,
            task=task,
            worker_name=getattr(self.worker, "name", self.worker.__class__.__name__),
        )

        result = self.worker.execute(task)
        if result.status == "failed":
            error_message = result.error or f"{self.worker.name} failed."
            trace_log.log_event(
                "research_expert_error",
                expert_id=self.id,
                role=self.role,
                result=result,
                error=error_message,
            )
            raise RuntimeError(error_message)
        note = self._result_to_note(result)
        trace_log.log_event(
            "research_expert_note",
            expert_id=self.id,
            role=self.role,
            result=result,
            note=note,
        )
        return note

    def _build_task(self) -> AgentTask:
        query = self.board.question
        if self.task_kind == "document_research" and self.document_path is not None:
            query = (
                f"Используя документ {self.document_path.name}, извлеки наиболее полезные для общей задачи "
                f"факты, приоритеты, меры, ограничения или числовые данные.\n"
                f"Общий вопрос команды: {self.board.question}"
            )

        context: dict[str, Any] = {
            "board_question": self.board.question,
            "board_notes": self.board.get_notes(),
            "expert_role": self.role.model_dump(),
        }
        metadata: dict[str, Any] = {
            "expert_id": self.id,
            "expert_role_name": self.role.name,
        }
        if self.document_path is not None:
            metadata["document_path"] = str(self.document_path)
            metadata["document_name"] = self.document_path.name

        constraints = {
            "audience": "blackboard",
            "return_shape": "base_note_ready",
            "focus": "Add concise, source-backed information that helps the team answer the board question.",
        }

        return AgentTask(
            kind=self.task_kind,
            query=query,
            context=context,
            constraints=constraints,
            metadata=metadata,
        )

    def _result_to_note(self, result: AgentResult) -> BaseNote:
        summary = self._truncate(result.summary.strip() or self.role.description, _SUMMARY_LIMIT)
        content = self._format_content(result)
        keywords = self._build_keywords(result)
        return BaseNote(
            content=self._truncate(content, _CONTENT_LIMIT),
            summary=summary,
            keywords=keywords[:_KEYWORD_LIMIT],
        )

    def _format_content(self, result: AgentResult) -> str:
        lines = [result.summary.strip()]
        if result.status == "partial":
            lines.append("Статус: предварительный вывод, требуется дополнительная проверка.")

        if result.sources:
            lines.append("Источники:")
            for source in result.sources[:_SOURCE_LIMIT]:
                parts = [f"- [{source.type}]"]
                label = (source.title or "").strip()
                if label:
                    parts.append(label)
                locator = (source.locator or "").strip()
                if locator:
                    parts.append(f"({locator})")
                snippet = self._truncate((source.snippet or "").strip(), 180)
                line = " ".join(part for part in parts if part).strip()
                if snippet:
                    line += f": {snippet}"
                lines.append(line)

        return "\n".join(line for line in lines if line)

    def _build_keywords(self, result: AgentResult) -> list[str]:
        keywords: list[str] = []
        for token in [self.task_kind.replace("_research", ""), "research"]:
            self._append_keyword(keywords, token)

        if self.document_path is not None:
            for token in re.split(r"[^A-Za-zА-Яа-яЁё0-9]+", self.document_path.stem):
                self._append_keyword(keywords, token)

        for source in result.sources:
            self._append_keyword(keywords, source.type)

        if not keywords:
            keywords.append("research")
        return keywords

    def _append_keyword(self, keywords: list[str], token: str) -> None:
        normalized = token.strip().lower()
        if not normalized or normalized in keywords:
            return
        keywords.append(normalized)

    def _truncate(self, value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return value[: limit - 3].rstrip() + "..."


def create_web_research_expert(board: Board, **kwargs: Any) -> TaskResearchExpertAdapter:
    role = Role(
        name="Веб-эксперт",
        description="Находит и кратко фиксирует подтвержденные сведения из web и Wikipedia.",
    )
    worker = create_web_research_agent(**kwargs)
    return TaskResearchExpertAdapter(
        board=board,
        worker=worker,
        role=role,
        task_kind="web_research",
    )


def create_document_research_experts(
    board: Board,
    storage_dir: str | Path | None = None,
    **kwargs: Any,
) -> list[TaskResearchExpertAdapter]:
    experts: list[TaskResearchExpertAdapter] = []
    for document_path in iter_document_paths(storage_dir):
        role = Role(
            name=f"Документ-эксперт: {document_path.stem}",
            description=f"Анализирует документ {document_path.name} и добавляет подтвержденные выдержки на доску.",
        )
        store = Store(str(document_path))
        worker = create_document_research_agent(store=store, **kwargs)
        experts.append(
            TaskResearchExpertAdapter(
                board=board,
                worker=worker,
                role=role,
                task_kind="document_research",
                document_path=document_path,
            )
        )
    return experts


__all__ = [
    "TaskResearchExpertAdapter",
    "create_document_research_experts",
    "create_web_research_expert",
]
