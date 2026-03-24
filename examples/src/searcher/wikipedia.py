from typing import Literal
import re
from pydantic import BaseModel, Field
import wikipedia
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .. import embedding

class PageSummary(BaseModel):
    id : int = Field(description='ID страницы')
    title : str = Field(description='Заголовок страницы')
    summary : str = Field(description='Краткое содержание страницы')

class WikipediaSearcher():

    def __init__(self, chunk_size : int = 1000, chunk_overlap : int = 100):
        self.stores = {}
        self.embedding = embedding
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def _get_page_docs(self, page_id):
        page = wikipedia.page(pageid=page_id, auto_suggest=False)
        splits = page.content.split('\n\n\n')
        path = [page.title]
        docs = []
        for split in splits:
            matches = re.findall(r'^(=+)\s*(.+?)\s*=+', split)
            if len(matches)>0:
                h, heading = matches[0]
                i = len(h) - 1
                path = [*path[:i], heading]
                split = split.removeprefix(f'{h} {heading} {h}')
            doc = Document(split, metadata={'path':path})
            if len(doc.page_content) > 0:
                docs.append(doc)
        return self.splitter.split_documents(docs)

    def _store_page(self, page_id : int) -> InMemoryVectorStore:
        if page_id in self.stores:
            return self.stores[page_id]
        docs = self._get_page_docs(page_id)
        store = InMemoryVectorStore.from_documents(docs, self.embedding)
        self.stores[page_id] = store
        return store 

    def search_on_wiki_page(self, query : str, page_id : int) -> list:
        """
        Возвращает результаты поиска по выбранной странице Википедии

        Args:
        - query : str - текстовый запрос
        - page_id : int - id страницы на Википедии

        Notes:
        - page_id может быть получен из PageSummary в search_for_wiki_pages()
        - если ответ пустой, вероятно, page_id указан неверно
        """
        try:
            store = self._store_page(page_id)
            return store.similarity_search(query, k=5)
        except:
            return []

    def search_for_wiki_pages(self, query : str, lang : Literal['ru', 'en'] = 'ru') -> list[PageSummary]:
        """
        Возвращает до 3 первых страниц на Википедии по указанному запросу на выбранном языке.
        
        Args:
        - query : str - текстовый запрос
        - lang : Literal['ru', 'en'] - выбранный язык Википедии (по умолчанию 'ru')
        """
        wikipedia.set_lang(lang)
        results = wikipedia.search(query, results=3)
        pages = []
        for r in results:
            try:
                page = wikipedia.page(title=r, auto_suggest=True)
                pages.append(page)
            except:
                pass
        unique_pages = {p.pageid:p for p in pages}
        return [PageSummary(
            id=up.pageid,
            title=up.title,
            summary=up.summary
        ) for up in unique_pages.values()]
    
    @property
    def tools(self):
        return [
            tool(self.search_for_wiki_pages),
            tool(self.search_on_wiki_page)
        ]
    
