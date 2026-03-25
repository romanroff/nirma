from pydantic import BaseModel, Field, model_validator
from ..utils import get_id

class BaseNote(BaseModel):
    """
    Короткая заметка для добавления на доску
    """
    content : str = Field(max_length=2000, description='Содержимое записи')
    summary : str = Field(max_length=280, description='Суть записи в одном коротком предложении')
    keywords : list[str] = Field(min_length=1, max_length=5, description='Ключевые слова')

class Note(BaseNote):
    id : str = Field(default='', description='ID записи')
    author_id : str = Field(description='ID автора записи')
    author_role : str = Field(description='Роль автора записи')

    @model_validator(mode='after')
    def _validate_model(self):
        self.id = get_id()
        return self
    
__all__=[
    'BaseNote',
    'Note'
]