from pydantic import BaseModel, Field
from langchain.tools import tool
from ._prompts import SELF_PROMPT, BOARD_PROMPT
from ..role import Role, RoleAgent
from ...board import Board, BaseNote

DECIDER_PROMPT = """
Ваша задача проанализировать текущее состояние общей доски и решить, достаточно ли у команды информации для получения окончательного ответа.
Если информации на доске достаточно для решения задачи, вы должны указать, что работа завершена. 
Если для получения ответа необходима дополнительная информация от других агентов, вы должны указать, что процесс следует продолжить.
Не решайте задачу.
"""
ROLE_NAME = 'Арбитр'
ROLE_DESCRIPTION = 'Оценивает полноту информации. Останавливает либо инициирует продолжение обсуждения'

_role = Role(
    name=ROLE_NAME,
    description=ROLE_DESCRIPTION
)

class DeciderResponse(BaseModel):
    note : BaseNote = Field(description='Запись для добавления на доску')
    is_final : bool = Field(default=False, description='Сигнал о завершении процесса работы над задачей')

def create_decider_agent(board : Board):
    system_prompt = SELF_PROMPT + BOARD_PROMPT.format(question=board.question) + DECIDER_PROMPT
    agent = RoleAgent(
        role=_role,
        system_prompt=system_prompt,
        tools=board.tools,
        response_format=DeciderResponse,
    )
    return agent