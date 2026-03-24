from langchain.tools import tool
from ._prompts import SELF_PROMPT, BOARD_PROMPT
from ..role import Role, RoleAgent
from ...board import Board, BaseNote

EXPERT_PROMPT = """
Вы выдающийся специалист:
{role_name}
Описание:
{role_description}

Основываясь на ваших экспертных знаниях и текущем содержимом общей доски, решите задачу, изложите свои идеи и информацию, которую вы хотите записать на доску.
Совершенно не обязательно полностью соглашаться с точкой зрения, представленной на доске.
"""

def create_expert_agent(role : Role, board : Board):
    system_prompt = SELF_PROMPT + BOARD_PROMPT.format(question=board.question) + EXPERT_PROMPT
    agent = RoleAgent(
        role=role,
        system_prompt=system_prompt,
        tools=board.tools,
        response_format=BaseNote,
    )
    return agent