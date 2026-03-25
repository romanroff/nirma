import os
import warnings

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from . import trace_log
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


def _get_env(name: str, default: str) -> str:
    return os.getenv(name, default)


class LoggingChatOllama(ChatOllama):

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        trace_log.log_event(
            "ollama_chat_request",
            client="ChatOllama",
            model=getattr(self, "model", None),
            base_url=getattr(self, "base_url", None),
            messages=messages,
            stop=stop,
            kwargs=kwargs,
        )
        try:
            result = super()._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        except Exception as exc:
            trace_log.log_event(
                "ollama_chat_error",
                client="ChatOllama",
                model=getattr(self, "model", None),
                base_url=getattr(self, "base_url", None),
                error=exc,
            )
            raise
        trace_log.log_event(
            "ollama_chat_response",
            client="ChatOllama",
            model=getattr(self, "model", None),
            base_url=getattr(self, "base_url", None),
            response=result,
        )
        return result


llm = LoggingChatOllama(
    model=_get_env('CHAT_MODEL', 'gpt-oss:120b'),
    base_url=_get_env('BASE_URL', 'http://localhost:11434'),
    temperature=_get_temperature(),
)

agent_llm = RequestsToolChatModel(
    model=_get_env('CHAT_MODEL', 'gpt-oss:120b'),
    base_url=_get_env('BASE_URL', 'http://localhost:11434'),
    api_key=os.getenv('API_KEY', 'ollama'),
    temperature=_get_temperature(),
)

embedding = OllamaEmbeddings(
    model=_get_env('EMBEDDING_MODEL', 'nomic-embed-text:latest'),
    base_url=_get_env('BASE_URL', 'http://localhost:11434'),
)

__all__ = [
    'agent_llm',
    'llm',
    'embedding',
]
