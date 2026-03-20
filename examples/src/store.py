from pathlib import Path
from typing import Literal
from langchain.tools import tool
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from .llms import llm, embedding

def _filter_metadata(docs):
    for i,d in enumerate(docs):
        metadata = d.metadata
        
        for key in [
            'source',
            'coordinates',
            'file_directory',
            'filename',
            'filetype',
            'languages',
            'last_modified',
            'text_as_html',
            'emphasized_text_contents',
            'emphasized_text_tags',
            'orig_elements',
        ]:
            if key in metadata:
                del metadata[key]

        metadata['chunk_index'] = i

    return docs

class Store():

    def __init__(self, path : str, mode : str='elements'):
        self._path = path
        self._document_name = Path(path).name
        
        if '.pdf' in path:
            Loader = UnstructuredPDFLoader
        elif '.docx' in path:
            Loader = UnstructuredWordDocumentLoader
        else:
            raise ValueError('Формат файла не поддерживается')
        
        loader = Loader(
            path, 
            mode=mode, 
            languages=['rus', 'eng'],
            chunking_strategy="by_title",  # или "basic"
            max_characters=2000,  # Максимальный размер чанка
            new_after_n_chars=1500,  # "Мягкий" максимум
            overlap=200,  # Перекрытие между чанками
            combine_text_under_n_chars=500
        )
        docs = loader.load()
        _filter_metadata(docs)
        
        self._docs = docs
        self._store = InMemoryVectorStore.from_documents(docs, embedding=embedding)

    def search(
        self,
        query : str,
        search_type: Literal['similarity', 'mmr']='similarity',
        k : int = 10,
        filter : dict | None = None,
        retriever : Literal['mq', 'cc', None] = None,
    ) -> list[Document]:
        base_retriever = self._store.as_retriever(
            search_type=search_type,
            search_kwargs={
                'k': k,
                'filter': self._get_filter(filter or {})
            }
        )
        if retriever == 'mq':
            final_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm
            )
        elif retriever == 'cc':
            final_retriever = ContextualCompressionRetriever(
                base_retriever=base_retriever,
                base_compressor=LLMChainExtractor.from_llm(llm)
            )
        else:
            final_retriever = base_retriever

        return final_retriever.invoke(query)

    def _serialize_docs(self, docs : list[Document]) -> list[dict]:
        results = []
        for doc in docs:
            metadata = dict(doc.metadata)
            page_number = metadata.get('page_number')
            chunk_index = metadata.get('chunk_index')
            locator = self._document_name
            if page_number is not None:
                locator += f'#page={page_number}'
            if chunk_index is not None:
                locator += f'&chunk={chunk_index}'

            results.append({
                'type': 'document',
                'title': self._document_name,
                'locator': locator,
                'snippet': doc.page_content[:500],
                'metadata': metadata,
                'content': doc.page_content,
            })
        return results
    
    def _get_filter(self, filter : dict):

        def _filter(doc) -> bool:
            b = True
            for k,f in filter.items():
                if k in doc.metadata:
                    value = doc.metadata[k]
                    for o,v in f.items():
                        if o == '<=':
                            b = b and (value <= v)
                        elif o == '==':
                            b = b and (value == v)
                        elif o == '>=':
                            b = b and (value >= v) 
            return b

        return _filter

    @property
    def tool(self):

        def search(query : str, search_type: Literal['similarity', 'mmr'], k : int = 10, filter : dict | None = None, retriever : Literal['mq', 'cc', None] = None) -> dict:
            """
            Поиск информации по текущему документу в векторной базе данных.
            
            Args:
                query: Поисковый запрос. Может быть пустой строкой
                search_type: Тип поиска ('similarity' или 'mmr')
                    - 'similarity' - семантический поиск
                    - 'mmr' - поиск с разнообразием (maximal marginal relevance)
                k: Количество возвращаемых результатов (по умолчанию 10)
                filter: Фильтр по ключам metadata (логическое "И" между ключами). Доступны фильтры '>=', '<=', '==':
                    - {'page_number':{'>=':50, '<=':70}, 'parent_id':{'==': 'foobar'}} - вернет результаты между страницами 50 и 70, parent_id которых совпадает с 'foobar'
                    - {'page_number':{'==':50}} - вернет результаты на 50 странице
                retriever : Тип retriever ('mq', 'cc' или None)
                    - 'mq' (multi-query retriever)
                        Генерирует несколько вариантов запроса и ищет по каждому.
                        Использовать, когда:
                        - Запрос сформулирован неточно или есть синонимы
                        - Важно ничего не пропустить (высокий recall)
                        - Есть риск, что одним запросом не всё не найти 
                    - 'cc' (contextual compression retriever)
                        Сначала находит документы обычным поиском, затем сжимает их,
                        оставляя только релевантные фрагменты.
                        Использовать, когда:
                        - Документы очень длинные и содержат много "воды"
                        - Нужно сэкономить токены в промпте
                        - Важно отсеять шум и оставить только суть (высокая precision)
                    - None (обычный retriever)
                        Обычный векторный поиск без дополнительной обработки.
                        Использовать, когда:
                        - Запрос точен и однозначен
                        - Документы уже короткие и чистые
                        - Скорость важнее качества
                        - Не нужно ни расширять запрос, ни сжимать результаты
            """
        
            docs = self.search(
                query=query,
                search_type=search_type,
                k=k,
                filter=filter,
                retriever=retriever,
            )

            return {
                'results': docs 
            }
        
        return tool(search)

    @property
    def research_tool(self):

        def document_search(query : str, search_type: Literal['similarity', 'mmr']='similarity', k : int = 10, filter : dict | None = None, retriever : Literal['mq', 'cc', None] = None) -> dict:
            """
            Search the current document store and return normalized serialized results.
            """
            docs = self.search(
                query=query,
                search_type=search_type,
                k=k,
                filter=filter,
                retriever=retriever,
            )
            return {
                'results': self._serialize_docs(docs)
            }

        return tool(document_search)

__all__=[
    'Store'
]
    
