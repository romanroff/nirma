from rich.console import Group

from src.board import Board
from src.board import Note


def test_board_build_panel_content_returns_rich_group():
    board = Board(question="Q")
    note = Note(
        author_id="a1",
        author_role="Planner",
        summary="**Кратко:** важный вывод",
        content="# Заголовок\n\n- пункт 1\n- пункт 2",
        keywords=["plan", "city"],
    )

    renderable = board._build_panel_content(note)

    assert isinstance(renderable, Group)
