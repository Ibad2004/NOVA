import os

import ollama
from dotenv import load_dotenv

from app.llm.base import BaseLLMClient, LLMResponse, ToolCall

load_dotenv()


class OllamaClient(BaseLLMClient):
    provider = "ollama"

    def __init__(self, client=None):
        self.model = os.getenv("OLLAMA_MODEL", "qwen3:8b")

        self.base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

        self.temperature = float(
            os.getenv("OLLAMA_TEMPERATURE", "0.2")
        )

        self.max_tokens = int(
            os.getenv("OLLAMA_MAX_TOKENS", "2048")
        )

        # Allow injecting a mock SDK client for tests.
        self.client = client or ollama.Client(host=self.base_url)

    @staticmethod
    def to_ollama_tools(tools: list[dict] | None) -> list[dict]:
        """Convert NOVA tool definitions to Ollama's format."""

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
    def to_ollama_messages(messages: list[dict]) -> list[dict]:
        """Convert NOVA messages to Ollama's format."""

        converted = []

        for message in messages:
            role = message["role"]
            content = message.get("content", "") or ""

            if role == "assistant":
                ollama_message = {"role": "assistant", "content": content}
                tool_calls = message.get("tool_calls") or []

                if tool_calls:
                    ollama_message["tool_calls"] = [
                        {
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            }
                        }
                        for call in tool_calls
                    ]

                converted.append(ollama_message)

            elif role == "tool":
                converted.append({
                    "role": "tool",
                    "content": content,
                    "tool_name": message.get("name", ""),
                })

            else:
                converted.append({"role": role, "content": content})

        return converted

    @staticmethod
    def from_ollama_response(response) -> LLMResponse:
        """Convert an Ollama chat response to an LLMResponse.

        Works with the SDK's ChatResponse objects and with plain dicts.
        """

        message = response["message"]
        tool_calls = [
            ToolCall(
                name=call["function"]["name"],
                arguments=dict(call["function"]["arguments"] or {}),
                id=f"call_{index}",
            )
            for index, call in enumerate(
                message.get("tool_calls") or []
            )
        ]

        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        think: bool = False,
    ) -> LLMResponse:
        request = {
            "model": self.model,
            "messages": self.to_ollama_messages(messages),
            "think": think,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        if tools:
            request["tools"] = self.to_ollama_tools(tools)

        response = self.client.chat(**request)

        return self.from_ollama_response(response)
