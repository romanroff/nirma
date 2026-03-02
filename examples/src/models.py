from pydantic import BaseModel, Field
from abc import ABC
import json

class Model(ABC, BaseModel):

    def __str__(self):
        schema = self.model_json_schema()
        data = self.model_dump()
        return json.dumps({
            'schema': schema,
            'data': data
        })
    
class QueryModel(Model):
    """
    Информация о запросе пользователя по разработке генерального плана населенного пункта
    """
    settlement_name : str = Field(description="Название населенного пункта")
    settlement_type : str = Field(description='Тип населенного пункта')
    current_year : int = Field(description='Текущий год')

__all__ = [
    'Model',
    'QueryModel'
]