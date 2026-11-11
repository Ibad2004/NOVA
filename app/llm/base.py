"""Provider-independent LLM interface for NOVA.

Every LLM client takes and returns NOVA's own formats, so the rest of
the code never deals with provider-specific shapes. Each client converts
to and from its provider's format internally.

Message format (``messages`` is a list of dicts, in conversation order):

    {"role": "system", "content": str}
    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": list[ToolCall]}
    {"role": "tool", "content": str, "tool_call_id": str, "name": str}

    - An assistant message may have empty ``content`` when it only
      requests tools. ``tool_calls`` may be omitted or empty.
    - A tool message answers one ToolCall: ``tool_call_id`` is that
      call's ``id`` and ``name`` is the tool's name.

Tool format (``tools`` is a list of dicts):

    {"name": str, "description": str, "parameters": dict}

    - ``parameters`` is a JSON Schema object describing the arguments.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    name: str
    arguments: dict
    id: str
    # Opaque provider data that must be sent back unchanged with this
    # call in later requests (e.g. Gemini's thought_signature).
    provider_data: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """A model reply: final text, tool calls, or both."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class BaseLLMClient(ABC):
    """Interface every LLM provider client must implement."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        think: bool = False,
    ) -> LLMResponse:
        """Send a conversation (and optional tools) to the model."""

    def ask(self, message: str, think: bool = False) -> str:
        """Send a single user prompt and return the reply text."""

        response = self.chat(
            [{"role": "user", "content": message}],
            think=think,
        )
        return response.content
