import json
import re
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ValidationError
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import BaseMessage
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from ..llms import agent_llm, llm
from .. import trace_log
from ..utils import get_id

STRUCTURED_RESPONSE_KEY = 'structured_response'
MESSAGES_KEY = 'messages'

class Agent():

    def __init__(
        self,
        *args,
        id_ : str | None = None,
        model : BaseChatModel | None = None,
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
        self.model = model or agent_llm
        self.response_format = response_format
        self.system_prompt = self._format_system_prompt(system_prompt)
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
            model=self.model, 
            tools=tools, 
            response_format=None, 
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            middleware=[summarization_middleware],
            **kwargs
        )

    def _format_system_prompt(self, system_prompt : str | None) -> str | None:
        if system_prompt is None or self.response_format is None:
            return system_prompt

        schema = json.dumps(
            self.response_format.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
        return (
            system_prompt.strip() + "\n\n"
            "Формат ответа:\n"
            "- Верни только валидный JSON.\n"
            "- Не используй markdown fences.\n"
            "- Строго следуй JSON Schema ниже.\n"
            f"{schema}"
        )

    def _parse_structured_response(self, response):
        if STRUCTURED_RESPONSE_KEY in response:
            return response[STRUCTURED_RESPONSE_KEY]

        if self.response_format is None:
            return response[MESSAGES_KEY][-1].content

        messages = response.get(MESSAGES_KEY) or []
        if not messages:
            raise ValueError('Agent returned no messages.')

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            raise ValueError('Agent did not finish with an AI message.')

        content = last_message.content
        if isinstance(content, list):
            content = '\n'.join(
                block.get('text', '')
                for block in content
                if isinstance(block, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Agent returned an empty final message.')

        parsed = self._extract_json_object(content)
        try:
            return self.response_format.model_validate_json(parsed)
        except ValidationError:
            repaired = self._repair_structured_payload(parsed)
            return self.response_format.model_validate(repaired)

    def _repair_structured_payload(self, parsed: str) -> Any:
        payload = json.loads(parsed)
        return self._coerce_model_payload(self.response_format, payload)

    def _coerce_model_payload(self, model_cls: type[BaseModel], payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload

        repaired = dict(payload)
        for field_name, field in model_cls.model_fields.items():
            if field_name not in repaired:
                continue
            repaired[field_name] = self._coerce_field_value(
                repaired[field_name],
                field.annotation,
                field,
            )
        return repaired

    def _coerce_field_value(self, value: Any, annotation: Any, field=None) -> Any:
        if value is None:
            return value

        model_cls = self._resolve_model_class(annotation)
        if model_cls is not None:
            return self._coerce_model_payload(model_cls, value)

        origin = get_origin(annotation)
        if origin in {list, tuple, set} and isinstance(value, list):
            item_annotation = get_args(annotation)[0] if get_args(annotation) else Any
            items = [
                self._coerce_field_value(item, item_annotation)
                for item in value
            ]
            max_length = self._get_constraint(field, 'max_length')
            if max_length is not None:
                items = items[:max_length]
            return items

        if annotation is str and isinstance(value, str):
            max_length = self._get_constraint(field, 'max_length')
            if max_length is None or len(value) <= max_length:
                return value
            return value[: max_length - 3].rstrip() + '...'

        return value

    def _resolve_model_class(self, annotation: Any) -> type[BaseModel] | None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation

        for candidate in get_args(annotation):
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                return candidate
        return None

    def _get_constraint(self, field, attr: str) -> int | None:
        if field is None:
            return None
        for metadata in getattr(field, 'metadata', []):
            if hasattr(metadata, attr):
                return getattr(metadata, attr)
        return None

    def _extract_json_object(self, content: str) -> str:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return cleaned

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1]
        raise ValueError(f"Could not parse JSON response from agent: {content[:200]}")

    def invoke(self, messages : BaseMessage, thread_id : str | None = None, force : bool = False):
        response = None
        trace_log.log_event(
            "agent_invoke_start",
            agent_id=self.id,
            agent_class=self.__class__.__name__,
            agent_role=getattr(getattr(self, "role", None), "name", None),
            thread_id=thread_id or self.id,
            force=force,
            response_format=getattr(self.response_format, "__name__", None),
            tools=[getattr(tool, "name", repr(tool)) for tool in self.tools],
            input_messages=messages,
        )

        def invoke():
            return self._agent.invoke(
                input={'messages': messages},
                config={"configurable": {"thread_id": thread_id or self.id}},
            )

        if force:
            while response is None:
                try:
                    response = invoke()
                except Exception as exc:
                    trace_log.log_event(
                        "agent_invoke_retry",
                        agent_id=self.id,
                        agent_class=self.__class__.__name__,
                        error=exc,
                    )
                    print('Maybe tool calls or something idk')
        else:
            try:
                response = invoke()
            except Exception as exc:
                trace_log.log_event(
                    "agent_invoke_error",
                    agent_id=self.id,
                    agent_class=self.__class__.__name__,
                    error=exc,
                )
                raise

        trace_log.log_event(
            "agent_invoke_raw_response",
            agent_id=self.id,
            agent_class=self.__class__.__name__,
            raw_response=response,
        )
        try:
            parsed_response = self._parse_structured_response(response)
        except Exception as exc:
            trace_log.log_event(
                "agent_invoke_parse_error",
                agent_id=self.id,
                agent_class=self.__class__.__name__,
                error=exc,
                raw_response=response,
            )
            raise

        trace_log.log_event(
            "agent_invoke_result",
            agent_id=self.id,
            agent_class=self.__class__.__name__,
            parsed_response=parsed_response,
        )
        return parsed_response
    
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
