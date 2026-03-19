from pydantic import BaseModel, model_validator
from abc import ABC
import json

class Model(ABC, BaseModel):

    @model_validator(mode='after')
    def _process_strings(self) -> 'Model':

        def _process_value(value : str) -> str:
            if isinstance(value, str):
                value = value.replace('\u202f', ' ')
            return value
        
        attributes = self.model_dump().keys()

        for attr in attributes:
            value = getattr(self, attr)
            if isinstance(value, list):
                new_value = [_process_value(v) for v in value]
            elif isinstance(value, dict):
                new_value = {k:_process_value(v) for k,v in value.items()}
            elif isinstance(value, set):
                new_value = {_process_value(v) for v in value}
            else:
                new_value = _process_value(value)
            setattr(self, attr, new_value)

        return self

    # def __str__(self):
    #     schema = self.model_json_schema()
    #     data = self.model_dump_json()
    #     return str({
    #         'schema': schema,
    #         'data': data
    #     })
    
    def to(self, cls : type[BaseModel]):
        return cls(**self.model_dump())
    
__all__=[
    'Model'
]