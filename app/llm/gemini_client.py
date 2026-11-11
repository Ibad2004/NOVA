import logging
import os
import uuid

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.llm.base import BaseLLMClient, LLMResponse, ToolCall

load_dotenv()

logger = logging.getLogger(__name__)


def _reason_name(value) -> str:
    """Readable name for an SDK enum such as FinishReason."""

    return getattr(value, "value", None) or str(value)


class GeminiClient(BaseLLMClient):
    """LLM client for the Gemini API (google-genai SDK)."""

    def __init__(self, client=None):
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        self.temperature = float(
            os.getenv("GEMINI_TEMPERATURE", "0.2")
        )

        self.max_tokens = int(
            os.getenv("GEMINI_MAX_TOKENS", "8192")
        )

        # Thinking budget used when think=True; -1 means dynamic.
        self.thinking_budget = int(
            os.getenv("GEMINI_THINKING_BUDGET") or "-1"
        )

        if client is None:
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Add it to your .env "
                    "file to use the Gemini provider."
                )

            client = genai.Client(api_key=api_key)

        # Allow injecting a mock SDK client for tests.
        self.client = client

    @staticmethod
    def to_gemini_tools(tools: list[dict] | None) -> list[types.Tool] | None:
        """Convert NOVA tool definitions to a single Gemini Tool."""

        if not tools:
            return None

        declarations = []

        for tool in tools:
            # Gemini rejects the JSON Schema "$schema" keyword.
            parameters = {
                key: value
                for key, value in (tool.get("parameters") or {}).items()
                if key != "$schema"
            }

            declarations.append(
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters_json_schema=parameters,
                )
            )

        return [types.Tool(function_declarations=declarations)]

    @staticmethod
    def to_gemini_contents(
        messages: list[dict],
    ) -> tuple[str | None, list[types.Content]]:
        """Convert NOVA messages to (system_instruction, contents)."""

        system_parts = []
        contents = []

        for message in messages:
            role = message["role"]
            content = message.get("content", "") or ""

            if role == "system":
                system_parts.append(content)

            elif role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=content)],
                    )
                )

            elif role == "assistant":
                parts = []

                if content:
                    parts.append(types.Part(text=content))

                for call in message.get("tool_calls") or []:
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                id=call.id,
                                name=call.name,
                                args=call.arguments,
                            ),
                            thought_signature=call.provider_data.get(
                                "thought_signature"
                            ),
                        )
                    )

                if parts:
                    contents.append(
                        types.Content(role="model", parts=parts)
                    )

            elif role == "tool":
                part = types.Part(
                    function_response=types.FunctionResponse(
                        id=message.get("tool_call_id"),
                        name=message.get("name", ""),
                        response={"result": content},
                    )
                )

                # Gemini expects all results for one model turn
                # together in a single user Content.
                previous = contents[-1] if contents else None
                if (
                    previous is not None
                    and previous.role == "user"
                    and previous.parts
                    and all(p.function_response for p in previous.parts)
                ):
                    previous.parts.append(part)
                else:
                    contents.append(
                        types.Content(role="user", parts=[part])
                    )

            else:
                raise ValueError(f"Unknown message role: {role!r}")

        system_instruction = (
            "\n\n".join(system_parts) if system_parts else None
        )
        return system_instruction, contents

    @staticmethod
    def from_gemini_response(response) -> LLMResponse:
        """Convert a Gemini response to an LLMResponse."""

        candidates = response.candidates or []
        if not candidates:
            feedback = response.prompt_feedback
            if feedback is not None:
                logger.warning(
                    "Gemini returned no candidates (prompt_feedback: "
                    "block_reason=%s, message=%s)",
                    _reason_name(feedback.block_reason),
                    feedback.block_reason_message,
                )
            else:
                logger.warning("Gemini returned no candidates.")
            return LLMResponse()

        candidate = candidates[0]
        parts = (
            candidate.content.parts
            if candidate.content is not None
            else None
        ) or []

        texts = []
        tool_calls = []

        for part in parts:
            # Thought summaries are internal reasoning, not the answer.
            if part.thought:
                continue

            if part.function_call:
                call = part.function_call
                provider_data = {}

                if part.thought_signature:
                    provider_data["thought_signature"] = (
                        part.thought_signature
                    )

                tool_calls.append(
                    ToolCall(
                        name=call.name,
                        arguments=dict(call.args or {}),
                        id=call.id or f"call_{uuid.uuid4().hex[:12]}",
                        provider_data=provider_data,
                    )
                )

            elif part.text:
                texts.append(part.text)

        if not texts and not tool_calls:
            # Usually MAX_TOKENS (thinking used up the budget) or SAFETY.
            logger.warning(
                "Gemini returned no text and no tool calls "
                "(finish_reason=%s).",
                _reason_name(candidate.finish_reason),
            )

        return LLMResponse(
            content="".join(texts),
            tool_calls=tool_calls,
        )

    def _thinking_config(self, think: bool) -> types.ThinkingConfig | None:
        """Map NOVA's think flag to a Gemini thinking budget.

        Thinking tokens count toward max_output_tokens, so leaving it
        on for simple requests can use up the budget before the model
        writes an answer.

        - think=True: dynamic budget (-1), or GEMINI_THINKING_BUDGET.
        - think=False: budget 0 on Flash models. Pro models cannot
          turn thinking off fully, so their default is kept.
        """

        if think:
            return types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )

        if "flash" in self.model.lower():
            return types.ThinkingConfig(thinking_budget=0)

        return None

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        think: bool = False,
    ) -> LLMResponse:
        system_instruction, contents = self.to_gemini_contents(messages)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            tools=self.to_gemini_tools(tools),
            # NOVA runs the tool loop itself, with permission checks.
            automatic_function_calling=(
                types.AutomaticFunctionCallingConfig(disable=True)
            ),
            thinking_config=self._thinking_config(think),
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return self.from_gemini_response(response)
