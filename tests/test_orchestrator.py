
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types as genai_types

from app.core.orchestrator import Orchestrator, _is_error_result
from app.llm.base import BaseLLMClient, LLMResponse, ToolCall
from app.llm.gemini_client import GeminiClient
from app.llm.groq_client import GroqClient
from app.llm.ollama_client import OllamaClient


# --------------------------------------------------
# Manual integration test
# Run with: python -m tests.test_orchestrator
# --------------------------------------------------

async def main():
    nova = Orchestrator()

    tools = await nova.mcp.list_tools()

    print("\nMCP TOOLS:")

    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    response = await nova.handle(
        "Tell me the names of the files and folders in my current folder."
    )

    print("\nNOVA:")
    print(response)


# --------------------------------------------------
# Error detection unit tests
# --------------------------------------------------

def test_detects_file_not_found():
    assert _is_error_result("ERROR: File not found: missing.py")


def test_detects_permission_denied():
    assert _is_error_result("Access denied: sensitive file.")


def test_accepts_successful_result():
    assert _is_error_result("PYTHON_SYNTAX_VALID") is False


def test_accepts_normal_file_content():
    assert _is_error_result("Hello, world!") is False


# --------------------------------------------------
# Mock tool results for retry tests
# --------------------------------------------------

def make_tool_result(text, is_error=False):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        is_error=is_error,
    )


# --------------------------------------------------
# Tool execution and retry tests
# --------------------------------------------------

def test_successful_tool_runs_once(monkeypatch):
    nova = Orchestrator()
    nova.max_tool_retries = 2

    mock_call = AsyncMock(
        return_value=make_tool_result("FILE_READ_SUCCESS")
    )
    monkeypatch.setattr(nova.mcp, "call_tool", mock_call)

    result = asyncio.run(
        nova._execute_tool_with_retries(
            "read_file",
            {"path": "example.txt"},
        )
    )

    assert result == "FILE_READ_SUCCESS"
    assert mock_call.await_count == 1


def test_tool_error_retries_and_then_succeeds(monkeypatch):
    nova = Orchestrator()
    nova.max_tool_retries = 2

    mock_call = AsyncMock(
        side_effect=[
            make_tool_result("ERROR: Temporary failure"),
            make_tool_result("FILE_READ_SUCCESS"),
        ]
    )
    monkeypatch.setattr(nova.mcp, "call_tool", mock_call)

    result = asyncio.run(
        nova._execute_tool_with_retries(
            "read_file",
            {"path": "example.txt"},
        )
    )

    assert result == "FILE_READ_SUCCESS"
    assert mock_call.await_count == 2


def test_tool_stops_after_retry_limit(monkeypatch):
    nova = Orchestrator()
    nova.max_tool_retries = 2

    mock_call = AsyncMock(
        return_value=make_tool_result("ERROR: File not found")
    )
    monkeypatch.setattr(nova.mcp, "call_tool", mock_call)

    result = asyncio.run(
        nova._execute_tool_with_retries(
            "read_file",
            {"path": "missing.txt"},
        )
    )

    assert "failed after 3 attempts" in result
    assert mock_call.await_count == 3


# --------------------------------------------------
# Fake MCP server for full tool-loop tests
# --------------------------------------------------

class FakeMCP:
    """In-memory stand-in for MCPClient with a single read_file tool."""

    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def list_tools(self):
        return [
            SimpleNamespace(
                name="read_file",
                description="Read a text file from the workspace.",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return make_tool_result("FILE_READ_SUCCESS\nhello")


FINAL_ANSWER = "The file says hello."


# --------------------------------------------------
# Provider clients with mocked SDKs
# Each scripts: read_file({"path": "a.txt"}), then a final answer.
# --------------------------------------------------

def make_ollama_llm():
    sdk = MagicMock()
    sdk.chat.side_effect = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "a.txt"},
                        }
                    }
                ],
            }
        },
        {"message": {"role": "assistant", "content": FINAL_ANSWER}},
    ]
    return OllamaClient(client=sdk)


def make_gemini_llm():
    def response(*parts):
        return genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(
                        role="model", parts=list(parts)
                    )
                )
            ]
        )

    sdk = MagicMock()
    sdk.models.generate_content.side_effect = [
        response(
            genai_types.Part(
                function_call=genai_types.FunctionCall(
                    id="fc_1",
                    name="read_file",
                    args={"path": "a.txt"},
                ),
                thought_signature=b"signature",
            )
        ),
        response(genai_types.Part(text=FINAL_ANSWER)),
    ]
    return GeminiClient(client=sdk)


def make_groq_llm():
    def response(content=None, tool_calls=None):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                    )
                )
            ]
        )

    sdk = MagicMock()
    sdk.chat.completions.create.side_effect = [
        response(tool_calls=[
            SimpleNamespace(
                id="call_abc",
                type="function",
                function=SimpleNamespace(
                    name="read_file",
                    arguments='{"path": "a.txt"}',
                ),
            )
        ]),
        response(content=FINAL_ANSWER),
    ]
    return GroqClient(client=sdk)


# --------------------------------------------------
# Full tool-loop tests
# --------------------------------------------------

@pytest.mark.parametrize(
    "make_llm",
    [make_ollama_llm, make_gemini_llm, make_groq_llm],
    ids=["ollama", "gemini", "groq"],
)
def test_tool_loop_works_with_each_provider(make_llm):
    nova = Orchestrator(llm=make_llm())
    fake_mcp = FakeMCP()
    nova.mcp = fake_mcp

    answer = asyncio.run(nova.handle("What is in a.txt?"))

    assert answer == FINAL_ANSWER
    assert fake_mcp.calls == [("read_file", {"path": "a.txt"})]


class ScriptedLLM(BaseLLMClient):
    """Returns scripted replies and records the messages it receives."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.received = []

    def chat(self, messages, tools=None, think=False):
        self.received.append([dict(message) for message in messages])
        return self.replies.pop(0)


def test_unauthorized_tool_is_denied_without_calling_mcp():
    llm = ScriptedLLM([
        LLMResponse(tool_calls=[
            ToolCall(name="delete_everything", arguments={},
                     id="call_bad"),
        ]),
        LLMResponse(content="I am not allowed to do that."),
    ])
    nova = Orchestrator(llm=llm)
    fake_mcp = FakeMCP()
    nova.mcp = fake_mcp

    answer = asyncio.run(nova.handle("Delete everything."))

    assert answer == "I am not allowed to do that."
    assert fake_mcp.calls == []

    tool_message = llm.received[1][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_bad"
    assert "Permission denied" in tool_message["content"]


class FailingLLM(BaseLLMClient):
    """Simulates a provider outage on every call."""

    provider = "gemini"

    def chat(self, messages, tools=None, think=False):
        raise ConnectionError("503 Service Unavailable\nretry later")


def test_llm_failure_returns_clear_message(caplog):
    nova = Orchestrator(llm=FailingLLM())
    fake_mcp = FakeMCP()
    nova.mcp = fake_mcp

    answer = asyncio.run(nova.handle("What is in a.txt?"))

    assert answer == (
        "NOVA could not reach the gemini model: "
        "ConnectionError: 503 Service Unavailable. "
        "Try again shortly or switch LLM_PROVIDER."
    )
    assert fake_mcp.calls == []
    assert "gemini" in caplog.text


def test_llm_failure_without_provider_name():
    class Broken(BaseLLMClient):
        def chat(self, messages, tools=None, think=False):
            raise RuntimeError("x" * 500)

    nova = Orchestrator(llm=Broken())
    nova.mcp = FakeMCP()

    answer = asyncio.run(nova.handle("Hi"))

    assert answer.startswith(
        "NOVA could not reach the unknown model: RuntimeError: xxx"
    )
    # The 500-character error message is cut to 200 characters.
    assert "x" * 200 + "..." in answer
    assert "x" * 201 not in answer


# --------------------------------------------------
# Manual test entry point
# --------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
