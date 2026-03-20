import os
import warnings

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from .tool_llm import RequestsToolChatModel

load_dotenv()

warnings.filterwarnings(
    "ignore",
    message=".*ChatOllama.*deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*OllamaEmbeddings.*deprecated.*",
)


def _get_temperature() -> float:
    return float(os.getenv('TEMPERATURE', '0.0'))


llm = ChatOllama(
    model=os.getenv('CHAT_MODEL'),
    base_url=os.getenv('BASE_URL'),
    temperature=_get_temperature(),
)

agent_llm = RequestsToolChatModel(
    model=os.getenv('CHAT_MODEL'),
    base_url=os.getenv('BASE_URL'),
    api_key=os.getenv('API_KEY', 'ollama'),
    temperature=_get_temperature(),
)

embedding = OllamaEmbeddings(
    model=os.getenv('EMBEDDING_MODEL'), 
    base_url=os.getenv('BASE_URL'),
)

__all__ = [
    'agent_llm',
    'llm',
    'embedding',
]
