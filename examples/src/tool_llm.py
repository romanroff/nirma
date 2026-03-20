from __future__ import annotations

from typing import Any, Callable, Sequence

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from langchain_community.adapters.openai import convert_dict_to_message


class RequestsToolChatModel(BaseChatModel):
    model: str
    base_url: str
    api_key: str = "ollama"
    temperature: float = 0.0
    timeout: float | None = 600.0
    max_retries: int = 0
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "requests-openai-compatible-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": convert_to_openai_messages(messages),
            "temperature": self.temperature,
            **self.model_kwargs,
        }
        if stop:
            payload["stop"] = stop
        if "tools" in kwargs and kwargs["tools"] is not None:
            payload["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs and kwargs["tool_choice"] is not None:
            payload["tool_choice"] = kwargs["tool_choice"]
        if "parallel_tool_calls" in kwargs and kwargs["parallel_tool_calls"] is not None:
            payload["parallel_tool_calls"] = kwargs["parallel_tool_calls"]

        response = requests.post(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            body = response.text.strip()
            message = body[:500] if body else f"HTTP {response.status_code}"
            raise RuntimeError(message)

        data = response.json()
        choice = data["choices"][0]
        message = convert_dict_to_message(choice["message"])

        if isinstance(message, AIMessage):
            usage = data.get("usage")
            if usage:
                message.usage_metadata = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

        generation = ChatGeneration(
            message=message,
            generation_info={"finish_reason": choice.get("finish_reason")},
        )
        return ChatResult(
            generations=[generation],
            llm_output={"usage": data.get("usage", {})},
        )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> Runnable:
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        if tool_choice:
            tool_names = [
                tool["function"]["name"]
                for tool in formatted_tools
                if "function" in tool and "name" in tool["function"]
            ]
            if isinstance(tool_choice, str):
                if tool_choice in tool_names:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                elif tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            kwargs["tool_choice"] = tool_choice

        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls

        return super().bind(tools=formatted_tools, **kwargs)


__all__ = ["RequestsToolChatModel"]
