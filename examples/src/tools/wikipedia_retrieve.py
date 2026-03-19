from langchain_community.retrievers import WikipediaRetriever
from langchain.tools import tool

def wikipedia_retrieve(query : str, k : int = 3, lang : str = 'ru'):
    """
    Обращается к Википедии с запросом и возвращает релевантные документы.

    Args:
        query : str - текст запроса
        k : int - количество релевантных документов (по умолчанию 3)
        lang : str - язык википедии (по умолчанию 'ru')
    """
    retriever = WikipediaRetriever(top_k_results=k, lang=lang)
    return retriever.invoke(query)

wikipedia_retrieve_tool = tool(wikipedia_retrieve)