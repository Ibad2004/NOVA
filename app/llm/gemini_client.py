import os
import uuid

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.llm.base import BaseLLMClient, LLMResponse, ToolCall

load_dotenv()


class GeminiClient(BaseLLMClient):
    """LLM client for the Gemini API (google-genai SDK)."""

    def __init__(self, client=None):
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        self.temperature = float(
            os.getenv("GEMINI_TEMPERATURE", "0.2")
        )

        self.max_tokens = int(
            os.getenv("GEMINI_MAX_TOKENS", "2048")
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
            return LLMResponse()

        content = candidates[0].content
        if content is None or not content.parts:
            return LLMResponse()

        texts = []
        tool_calls = []

        for part in content.parts:
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

        return LLMResponse(
            content="".join(texts),
            tool_calls=tool_calls,
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        think: bool = False,
    ) -> LLMResponse:
        # `think` is ignored: Gemini 2.5+ models decide how much to
        # think on their own (dynamic thinking budget), so NOVA's
        # keyword-based router does not need to switch it on or off.
        # Accepting the flag keeps the BaseLLMClient interface the
        # same for every provider.
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
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return self.from_gemini_response(response)
