from pydantic import BaseModel, Field
from ..agent import Agent
from ..role import Role
from ...board import Board

GENERATOR_PROMPT = """
Вам дан вопрос. Предоставьте мне список из 1-{k_roles} экспертных ролей, которые наиболее полезны для решения вопроса.
Вопрос:
{question}
"""

class GeneratorResponse(BaseModel):
    roles : list[Role] = Field(min_length=1, description='Список экспертных ролей')

def create_generator_agent(board : Board, k_roles : int = 3) -> Agent:
    system_prompt = GENERATOR_PROMPT.format(question=board.question, k_roles=k_roles)
    return Agent(system_prompt=system_prompt, response_format=GeneratorResponse)
