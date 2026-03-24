from ._prompts import SELF_PROMPT, BOARD_PROMPT
from ...searcher.wikipedia import WikipediaSearcher
from ...board import Board, BaseNote
from ..role import Role, RoleAgent

WIKIPEDIA_PROMPT = """
Используя Википедию и основываясь на текущем содержимом общей доски, решите задачу, изложите свои идеи и информацию, которую вы хотите записать на доску.
Совершенно не обязательно полностью соглашаться с точкой зрения, представленной на доске.

Правила работы:
- Не вызывай один и тот же инструмент доступа к Википедии повторно с почти одинаковыми запросами, если только предыдущий вызов не вернул ноль полезных результатов.
- После одного или двух полезных вызовов инструментов доступа к Википедии остановись и верни итоговый ответ.
- Не используй общие знания. Отвечай на основе данных из Википедии.
"""
ROLE_NAME = 'Вики-эксперт'
ROLE_DESCRIPTION = 'Находит и обобщает информацию из Википедии'

_role = Role(
    name=ROLE_NAME,
    description=ROLE_DESCRIPTION
)

def create_wikipedia_agent(searcher : WikipediaSearcher, board : Board):
    system_prompt = SELF_PROMPT + BOARD_PROMPT.format(question=board.question) + WIKIPEDIA_PROMPT
    agent = RoleAgent(
        role=_role,
        system_prompt=system_prompt,
        tools=[
            *board.tools,
            *searcher.tools
        ],
        response_format=BaseNote,
    )
    return agent