from pydantic import BaseModel, Field
from langchain.tools import tool
from .note import BaseNote, Note

class Board(BaseModel):
    question : str = Field(description='Вопрос пользователя')
    notes : list[BaseNote] = Field(default=[], description='Список записей')

    def get_notes(self, last_n : int | None = None) -> list[dict]:
        """
        Возвращает список актуальных записей на доске с краткой информацией.
        Если передан last_n, вернет последние last_n записей.
        """
        notes = [{
            'id': note.id,
            'author_id': note.author_id,
            'author_role': note.author_role,
            'summary': note.summary,
            'keywords': note.keywords
        } for note in self.notes]
        if last_n is not None:
            notes = notes[-last_n:]
        return notes
    
    def get_note(self, note_id : str) -> Note | None:
        """
        Возвращает запись с указанным id.
        Возвращает None, если запись не найдена.
        """
        notes = [n for n in self.notes if n.id == note_id]
        if len(notes) == 0:
            return None
        return notes[0]
    
    def add_note(self, note : BaseNote, author_id : str, author_role : str) -> str:
        """
        Добавляет запись на доску.
        Возвращает id записи.
        """
        note = Note(author_id=author_id, author_role=author_role, **note.model_dump())
        self.notes.append(note)
        return note.id
    
    def remove_note(self, note_id : str):
        self.notes = [n for n in self.notes if n.id != note_id]

    def remove_notes(self, notes_ids : list[str]):
        self.notes = [n for n in self.notes if n.id not in notes_ids]

    def _build_panel_content(self, note: Note):
        from rich.console import Group
        from rich.markdown import Markdown
        from rich.rule import Rule
        from rich.text import Text

        keywords = ', '.join(note.keywords)
        return Group(
            Markdown(note.summary),
            Rule(style='dim'),
            Markdown(note.content),
            Rule(style='dim'),
            Text(f'keywords: {keywords}', style='cyan'),
        )

    def print(self, color = 'yellow', width : int = 100):
        from rich import print
        from rich.panel import Panel

        for note in self.notes:
            panel_title = f"📌 {note.author_role} [{note.author_id}]"
            
            print(Panel(
                self._build_panel_content(note),
                title=panel_title,
                border_style=color,
                width=width
            ))

    @property
    def tools(self):
        return [
            tool(self.get_notes),
            tool(self.get_note)
        ]

__all__=[
    'Board'
]
