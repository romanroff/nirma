from pydantic import BaseModel
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import BaseMessage
from langchain.agents import create_agent
from ..llms import llm
from ..utils import get_id

STRUCTURED_RESPONSE_KEY = 'structured_response'
MESSAGES_KEY = 'messages'

class Agent():

    def __init__(
        self,
        *args,
        id_ : str | None = None,
        tools : list | None = None,
        system_prompt : str | None = None,
        response_format : type[BaseModel] | None = None,
        checkpointer : InMemorySaver | None = None,
        summarization_tokens : int = 4000, 
        summarization_keep : int = 2,
        **kwargs
    ):
        self.id = id_ or get_id()
        
        self.tools = tools or []
        self.system_prompt = self._format_system_prompt(system_prompt)
        self.response_format = response_format
        if response_format is not None: # FIXME CRITICAL
            import langgraph.checkpoint.serde
            langgraph.checkpoint.serde._msgpack.SAFE_MSGPACK_TYPES = (
                langgraph.checkpoint.serde._msgpack.SAFE_MSGPACK_TYPES.union({(response_format.__module__, response_format.__name__)})
            )
        self.checkpointer = checkpointer or InMemorySaver()

        summarization_middleware = SummarizationMiddleware(
            model=llm,
            trigger=("tokens", summarization_tokens),
            keep=("messages", summarization_keep)
        )
        
        self._agent = create_agent(
            *args,
            model=llm, 
            tools=tools, 
            response_format=response_format, 
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            middleware=[summarization_middleware],
            **kwargs
        )

    def _format_system_prompt(self, system_prompt : str | None) -> str | None:
        return system_prompt

    def invoke(self, messages : BaseMessage, thread_id : str | None = None, force : bool = False):
        response = None

        def invoke():
            return self._agent.invoke(
                input={'messages': messages},
                config={"configurable": {"thread_id": thread_id or self.id}},
            )

        if force:
            while response is None:
                try:
                    response = invoke()
                except:
                    print('Maybe tool calls or something idk')
        else:
            response = invoke()
            
        if STRUCTURED_RESPONSE_KEY in response:
            return response[STRUCTURED_RESPONSE_KEY]
        return response[MESSAGES_KEY][-1].content
    
    @property
    def info(self):
        return {
            'id': self.id
        }
    
    @property
    def messages(self) -> list:
        thread_id = self.id
        thread = self.checkpointer.get({"configurable": {"thread_id": thread_id}})
        return thread['channel_values']['messages']
    
__all__=[
    'Agent'
]
