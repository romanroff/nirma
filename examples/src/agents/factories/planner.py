from langchain.tools import tool
from ._prompts import SELF_PROMPT, BOARD_PROMPT
from ..role import Role, RoleAgent
from ...board import Board, BaseNote

PLANNER_PROMPT = """
Создайте план решения исходной задачи на основе текущего содержимого общей доски.
Опишите задачу своими словами, затем изложите пошаговый план её решения.
Если план уже существует на доске или задача достаточно проста для прямого решения, просто укажите, что нет необходимости декомпозировать задачи и вы ожидаете дополнительной информации.
Не решайте задачу. Предоставьте только план.
"""
ROLE_NAME = 'Планировщик'
ROLE_DESCRIPTION = 'Разрабатывает пошаговый план решения задачи на основе содержимого доски'

_role = Role(
    name=ROLE_NAME,
    description=ROLE_DESCRIPTION
)

def create_planner_agent(board : Board):
    system_prompt = SELF_PROMPT + BOARD_PROMPT.format(question=board.question) + PLANNER_PROMPT
    agent = RoleAgent(
        role=_role,
        system_prompt=system_prompt,
        tools=board.tools,
        response_format=BaseNote,
    )
    return agent