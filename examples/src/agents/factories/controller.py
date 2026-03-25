from pydantic import BaseModel, Field
from langchain.tools import tool
from ..agent import Agent
from ..role import RoleAgent
from ...board import Board

CONTROLLER_PROMPT = """
Ваша задача назначить других агентов для сотрудничества и решения данной задачи.
Имена агентов и их описания перечислены ниже:
{agents}
Данная задача:
{question}
Агенты обмениваются информацией через общую доску.
Основываясь на содержимом, которое уже есть на доске, вам необходимо выбрать подходящих агентов из списка, чтобы они оставили записи на доске.
"""

class ControllerResponse(BaseModel):
    agents_ids : list[str] = Field(min_length=1, description='Упорядоченный список ID агентов')

def create_controller_agent(role_agents : list[RoleAgent], board : Board):
    system_prompt = CONTROLLER_PROMPT.format(
        agents=[rl.info for rl in role_agents],
        question=board.question
    )
    return Agent(
        system_prompt=system_prompt,
        tools=board.tools,
        response_format=ControllerResponse,
    )
