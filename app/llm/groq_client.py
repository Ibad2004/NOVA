import json
import logging
import os

import groq
from dotenv import load_dotenv

from app.llm.base import BaseLLMClient, LLMResponse, ToolCall

load_dotenv()

logger = logging.getLogger(__name__)


class GroqClient(BaseLLMClient):
    """LLM client for Groq's OpenAI-compatible chat completions API."""

    provider = "groq"

    def __init__(self, client=None):
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        self.temperature = float(
            os.getenv("GROQ_TEMPERATURE", "0.2")
        )

        self.max_tokens = int(
            os.getenv("GROQ_MAX_TOKENS", "2048")
        )

        if client is None:
            api_key = os.getenv("GROQ_API_KEY")

            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to your .env "
                    "file to use the Groq provider."
                )

            client = groq.Groq(api_key=api_key)

        # Allow injecting a mock SDK client for tests.
        self.client = client

    @staticmethod
    def to_groq_tools(tools: list[dict] | None) -> list[dict]:
        """Convert NOVA tool definitions to Groq's format."""

        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }
            for tool in tools or []
        ]

    @staticmethod
    def to_groq_messages(messages: list[dict]) -> list[dict]:
        """Convert NOVA messages to Groq's format."""

        converted = []

        for message in messages:
            role = message["role"]
            content = message.get("content", "") or ""

            if role == "assistant":
                tool_calls = message.get("tool_calls") or []
                groq_message = {"role": "assistant", "content": content}

                if tool_calls:
                    # OpenAI-style APIs expect null content when the
                    # assistant only requested tools.
                    groq_message["content"] = content or None
                    groq_message["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in tool_calls
                    ]

                converted.append(groq_message)

            elif role == "tool":
                converted.append({
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": content,
                })

            else:
                converted.append({"role": role, "content": content})

        return converted

    @staticmethod
    def _parse_arguments(name: str, raw) -> dict:
        """Parse tool arguments sent by the model as a JSON string."""

        if isinstance(raw, dict):
            return raw

        if not raw:
            return {}

        try:
            arguments = json.loads(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid JSON arguments for tool %s: %r", name, raw
            )
            return {}

        if not isinstance(arguments, dict):
            logger.warning(
                "Tool %s arguments are not a JSON object: %r", name, raw
            )
            return {}

        return arguments

    @classmethod
    def from_groq_response(cls, response) -> LLMResponse:
        """Convert a Groq chat completion to an LLMResponse."""

        message = response.choices[0].message

        tool_calls = [
            ToolCall(
                name=call.function.name,
                arguments=cls._parse_arguments(
                    call.function.name, call.function.arguments
                ),
                id=call.id,
            )
            for call in message.tool_calls or []
        ]

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        think: bool = False,
    ) -> LLMResponse:
        # `think` is ignored: Groq's default models (e.g. Llama 3.3)
        # have no thinking mode, and the reasoning options Groq offers
        # only apply to specific reasoning models. Accepting the flag
        # keeps the BaseLLMClient interface the same for every provider.
        request = {
            "model": self.model,
            "messages": self.to_groq_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            request["tools"] = self.to_groq_tools(tools)
            request["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**request)

        return self.from_groq_response(response)
