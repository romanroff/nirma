from pydantic import BaseModel, Field
from langchain.tools import tool
from ._prompts import SELF_PROMPT, BOARD_PROMPT
from ..role import Role, RoleAgent
from ...board import Board, BaseNote

CRITIC_PROMPT = """
Проанализируйте записи на общей доске и определите любые бесполезные или избыточные.

Если вы обнаружите такие записи, опишите их и объясните, почему они должны быть удалены.
Если бесполезных записей нет, просто укажите, что бесполезных записей нет и вы ожидаете дополнительной информации.
"""
ROLE_NAME='Критик'
ROLE_DESCRIPTION = 'Выявляет ошибочные или вводящие в заблуждение записи на доске'

_role = Role(
    name=ROLE_NAME,
    description=ROLE_DESCRIPTION
)

def create_critic_agent(board : Board):
    system_prompt = SELF_PROMPT + BOARD_PROMPT.format(question=board.question) + CRITIC_PROMPT
    agent = RoleAgent(
        role=_role,
        system_prompt=system_prompt,
        tools=board.tools,
        response_format=BaseNote,
    )
    return agent