from pydantic import BaseModel, Field
from .agent import Agent
    
class Role(BaseModel):
    name : str = Field(description='Название роли')
    description : str = Field(max_length=140, description='Описание роли')
    
class RoleAgent(Agent):

    def __init__(self, role : Role, *args, **kwargs):
        self.role = role
        super().__init__(*args, **kwargs)

    def _format_system_prompt(self, system_prompt):
        if system_prompt is None:
            return system_prompt
        return system_prompt.format(**self.info)

    @property
    def info(self):
        return {
            **super().info,
            'role_name': self.role.name,
            'role_description': self.role.description
        }
    
__all__=[
    'Role',
    'RoleAgent'
]