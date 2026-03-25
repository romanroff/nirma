from pathlib import Path

import pytest

from src import AgentResult, AgentSource, Role, iter_document_paths
from src.agents.factories.research import (
    TaskResearchExpertAdapter,
    create_document_research_experts,
    create_web_research_expert,
)
from src.board import Board


class FakeWorker:
    def __init__(self, result, name: str = "fake_worker"):
        self._result = result
        self.name = name
        self.calls = []

    def execute(self, task):
        self.calls.append(task)
        return self._result


def test_iter_document_paths_filters_supported_extensions(tmp_path: Path):
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "plan.pdf").write_bytes(b"pdf")
    (storage / "notes.docx").write_bytes(b"docx")
    (storage / "ignore.txt").write_text("skip", encoding="utf-8")

    paths = iter_document_paths(storage)

    assert [path.name for path in paths] == ["notes.docx", "plan.pdf"]


def test_iter_document_paths_returns_empty_list_for_empty_storage(tmp_path: Path):
    storage = tmp_path / "storage"
    storage.mkdir()

    assert iter_document_paths(storage) == []


def test_web_research_adapter_converts_result_to_base_note():
    board = Board(question="Как развивать Гатчину?")
    worker = FakeWorker(
        AgentResult(
            task_id="task-1",
            agent_name="web_research_agent",
            status="success",
            summary="Гатчине стоит усилить транспорт и туризм.",
            sources=[
                AgentSource(
                    type="web",
                    title="Example",
                    locator="https://example.com",
                    snippet="Example snippet",
                    metadata={},
                )
            ],
            artifacts={},
            error=None,
        ),
        name="web_research_agent",
    )
    adapter = TaskResearchExpertAdapter(
        board=board,
        worker=worker,
        role=Role(
            name="Веб-эксперт",
            description="Находит сведения в сети.",
        ),
        task_kind="web_research",
    )

    note = adapter.invoke([])

    assert note.summary == "Гатчине стоит усилить транспорт и туризм."
    assert "Источники:" in note.content
    assert "https://example.com" in note.content
    assert note.keywords[0] == "web"
    assert worker.calls[0].context["board_question"] == board.question


def test_adapter_raises_on_failed_result():
    adapter = TaskResearchExpertAdapter(
        board=Board(question="Как развивать Гатчину?"),
        worker=FakeWorker(
            AgentResult(
                task_id="task-1",
                agent_name="document_research_agent",
                status="failed",
                summary="",
                sources=[],
                artifacts={},
                error="boom",
            ),
            name="document_research_agent",
        ),
        role=Role(name="Документ-эксперт", description="Работает с документом."),
        task_kind="document_research",
    )

    with pytest.raises(RuntimeError, match="boom"):
        adapter.invoke([])


def test_document_factory_returns_empty_list_for_empty_storage(tmp_path: Path):
    board = Board(question="Как развивать Гатчину?")

    experts = create_document_research_experts(board, storage_dir=tmp_path)

    assert experts == []


def test_document_factory_creates_one_expert_per_file(tmp_path: Path, monkeypatch):
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "plan.docx").write_bytes(b"docx")
    (storage / "transport.pdf").write_bytes(b"pdf")

    fake_stores = []
    fake_workers = []

    class FakeStore:
        def __init__(self, path: str):
            fake_stores.append(Path(path))
            self.path = path

    def fake_create_document_research_agent(store):
        worker = FakeWorker(
            AgentResult(
                task_id="task",
                agent_name="document_research_agent",
                status="success",
                summary=f"Summary for {Path(store.path).name}",
                sources=[],
                artifacts={},
                error=None,
            ),
            name="document_research_agent",
        )
        fake_workers.append(worker)
        return worker

    monkeypatch.setattr("src.agents.factories.research.Store", FakeStore)
    monkeypatch.setattr(
        "src.agents.factories.research.create_document_research_agent",
        fake_create_document_research_agent,
    )

    experts = create_document_research_experts(
        Board(question="Как развивать Гатчину?"),
        storage_dir=storage,
    )

    assert len(experts) == 2
    assert [expert.role.name for expert in experts] == [
        "Документ-эксперт: plan",
        "Документ-эксперт: transport",
    ]
    assert [path.name for path in fake_stores] == ["plan.docx", "transport.pdf"]


def test_public_factory_builds_web_expert(monkeypatch):
    worker = FakeWorker(
        AgentResult(
            task_id="task-1",
            agent_name="web_research_agent",
            status="success",
            summary="Summary",
            sources=[],
            artifacts={},
            error=None,
        ),
        name="web_research_agent",
    )

    monkeypatch.setattr(
        "src.agents.factories.research.create_web_research_agent",
        lambda **kwargs: worker,
    )

    expert = create_web_research_expert(Board(question="Q"))

    assert expert.role.name == "Веб-эксперт"
    assert expert.worker is worker


def test_document_factory_accepts_uppercase_docx_suffix(tmp_path: Path, monkeypatch):
    storage = tmp_path / "storage"
    storage.mkdir()
    document_path = storage / "UPPER.DOCX"
    document_path.write_bytes(b"docx")

    class FakeStore:
        def __init__(self, path: str):
            self.path = path

    monkeypatch.setattr("src.agents.factories.research.Store", FakeStore)
    monkeypatch.setattr(
        "src.agents.factories.research.create_document_research_agent",
        lambda store: FakeWorker(
            AgentResult(
                task_id="task",
                agent_name="document_research_agent",
                status="success",
                summary="ok",
                sources=[],
                artifacts={},
                error=None,
            ),
            name="document_research_agent",
        ),
    )

    experts = create_document_research_experts(
        Board(question="Как развивать Гатчину?"),
        storage_dir=storage,
    )

    assert len(experts) == 1
    assert experts[0].role.name == "Документ-эксперт: UPPER"


def test_document_adapter_builds_document_specific_query():
    board = Board(question="Как развивать Гатчину?")
    worker = FakeWorker(
        AgentResult(
            task_id="task-1",
            agent_name="document_research_agent",
            status="partial",
            summary="В документе есть релевантные приоритеты.",
            sources=[],
            artifacts={},
            error=None,
        ),
        name="document_research_agent",
    )
    adapter = TaskResearchExpertAdapter(
        board=board,
        worker=worker,
        role=Role(
            name="Документ-эксперт: plan",
            description="Работает с конкретным документом.",
        ),
        task_kind="document_research",
        document_path=Path("storage/plan.docx"),
    )

    adapter.invoke([])

    assert "Используя документ plan.docx" in worker.calls[0].query
    assert board.question in worker.calls[0].query
