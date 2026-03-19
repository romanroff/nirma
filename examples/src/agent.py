from langchain_core.messages import BaseMessage
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from .model import Model
from .llms import llm

STRUCTURED_RESPONSE_KEY = 'structured_response'
MESSAGES_KEY = 'messages'

class Agent():
    
    def __init__(self, *args, tools : list | None = None, system_prompt : str | None = None, response_format : type[Model] | None = None, summarization_tokens : int = 4000, summarization_keep : int = 10, **kwargs):
        checkpointer = InMemorySaver()
        middleware = SummarizationMiddleware(
            model=llm,
            trigger=("tokens", summarization_tokens),
            keep=("messages", summarization_keep),
        )
        self._agent = create_agent(
            *args,
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            response_format=response_format,
            middleware=[middleware],
            checkpointer=checkpointer,
            **kwargs
        )

    def run(self, messages : list[BaseMessage], **kwargs) -> str | Model:
        result = self._agent.invoke(
            input={MESSAGES_KEY: messages},
            config={"configurable": {"thread_id": "1"}},
            **kwargs
        )
        if STRUCTURED_RESPONSE_KEY in result:
            return result[STRUCTURED_RESPONSE_KEY]
        return result[MESSAGES_KEY][-1].content
    
__all__=[
    'Agent'
]