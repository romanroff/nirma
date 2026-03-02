import requests
from langchain.tools import tool
from typing import Literal
from langchain_openai import ChatOpenAI
from .const import API_KEY, LIGHTRAG_API

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.0,
)

# @tool
def _lightrag_retrieve(query: str, query_mode:Literal['local', 'global', 'hybrid', 'naive', 'mix'] = 'mix') -> str:
    """
    Отправляет запрос к API LightRAG и возвращает текстовый ответ.
    LightRAG хранит данные о документах стратегического и территориального планирования, а также градостроительных и нормативных документах.
    
    Parameters
    ----------
    query : str
        Текстовый запрос, который необходимо отправить в LightRAG.
        Может быть любой строкой, с которой модель или система должна работать.
    
    query_mode : Literal['local', 'global', 'hybrid', 'naive', 'mix'], optional
        Режим обработки запроса, который определяет стратегию поиска или генерации ответа:
        - 'local'  : Находит сначала ключевые сущности из знаний и строит контекст вокруг их непосредственных связей в графе.
        - 'global' : Опирается на связи между сущностями по всему графу, подбирая взаимосвязанные отношения (relationships), а затем соответствующие куски.
        - 'hybrid' : Комбинирует local + global подходы, т.е. одновременно локальные сущности и глобальные связи.
        - 'naive'  : Выполняет обычный векторным поиском по фрагментам текста (chunks) без использования графа знаний.
        - 'mix'    : Объединяет данные из графа знаний и обычный векторный поиск по фрагментам (по умолчанию).
    
    Returns
    -------
    str
        Текстовый ответ от API LightRAG.
    
    Example
    -------
    >>> result = lightrag_retrieve("Что такое стратегия социально-экономического развития?", query_mode='global')
    >>> print(result)
    'Стратегия социально-экономического развития...'
    """
    response = requests.post(
        f'{LIGHTRAG_API}/query',
        json={
            'query': query,
            'query_mode': query_mode
        }
    )
    return response.text

lightrag_retrieve = tool(_lightrag_retrieve)

__all__ = [
    'llm',
    'lightrag_retrieve'
]