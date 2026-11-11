import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.llm.base import LLMResponse, ToolCall
from app.llm.groq_client import GroqClient


def make_tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_client(content=None, tool_calls=None):
    """Build a GroqClient around a mock SDK client."""

    sdk = MagicMock()
    sdk.chat.completions.create.return_value = SimpleNamespace(
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
    return GroqClient(client=sdk), sdk.chat.completions.create


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqClient()


def test_model_defaults_and_env_settings(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setenv("GROQ_TEMPERATURE", "0.5")
    monkeypatch.setenv("GROQ_MAX_TOKENS", "100")

    client = GroqClient(client=MagicMock())

    assert client.model == "llama-3.3-70b-versatile"
    assert client.temperature == 0.5
    assert client.max_tokens == 100


def test_text_reply():
    client, create = make_client(content="Hello!")

    response = client.chat(
        [{"role": "user", "content": "Hi"}],
        think=True,
    )

    assert response == LLMResponse(content="Hello!", tool_calls=[])
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == client.model
    assert kwargs["temperature"] == client.temperature
    assert kwargs["max_tokens"] == client.max_tokens
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    assert "think" not in kwargs


def test_ask_returns_text():
    client, create = make_client(content="Islamabad")

    assert client.ask("Capital of Pakistan?") == "Islamabad"
    assert create.call_args.kwargs["messages"] == [
        {"role": "user", "content": "Capital of Pakistan?"}
    ]


def test_tools_sent_in_groq_format():
    client, create = make_client(content="ok")
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    client.chat(
        [{"role": "user", "content": "Read a.txt"}],
        tools=[{
            "name": "read_file",
            "description": "Read a file",
            "parameters": schema,
        }],
    )

    kwargs = create.call_args.kwargs
    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": schema,
            },
        }
    ]
    assert kwargs["tool_choice"] == "auto"


def test_tool_calls_converted_to_nova_format():
    client, _ = make_client(tool_calls=[
        make_tool_call("call_abc", "read_file", '{"path": "a.txt"}'),
        make_tool_call("call_def", "list_dir", "{}"),
    ])

    response = client.chat([{"role": "user", "content": "Go"}])

    assert response.content == ""
    assert response.tool_calls == [
        ToolCall(name="read_file", arguments={"path": "a.txt"},
                 id="call_abc"),
        ToolCall(name="list_dir", arguments={}, id="call_def"),
    ]


@pytest.mark.parametrize("raw", ["{not json", "[1, 2]", "", None])
def test_invalid_arguments_become_empty_dict(raw, caplog):
    client, _ = make_client(tool_calls=[
        make_tool_call("call_1", "read_file", raw),
    ])

    with caplog.at_level(logging.WARNING):
        response = client.chat([{"role": "user", "content": "Go"}])

    assert response.tool_calls[0].arguments == {}
    if raw:
        assert "read_file" in caplog.text


def test_history_converted_to_groq_format():
    client, create = make_client(content="Done")

    client.chat([
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Read a.txt"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                ToolCall(name="read_file", arguments={"path": "a.txt"},
                         id="call_abc"),
            ],
        },
        {
            "role": "tool",
            "content": "FILE_READ_SUCCESS",
            "tool_call_id": "call_abc",
            "name": "read_file",
        },
        {"role": "assistant", "content": "The file says hi."},
    ])

    assert create.call_args.kwargs["messages"] == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Read a.txt"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "a.txt"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": "FILE_READ_SUCCESS",
        },
        {"role": "assistant", "content": "The file says hi."},
    ]


def test_sdk_response_objects_are_supported():
    """The real SDK returns ChatCompletion objects."""

    from groq.types.chat import ChatCompletion

    sdk = MagicMock()
    sdk.chat.completions.create.return_value = ChatCompletion.model_validate({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "llama-3.3-70b-versatile",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "a.txt"}',
                    },
                }],
            },
        }],
    })

    response = GroqClient(client=sdk).chat(
        [{"role": "user", "content": "Go"}]
    )

    assert response == LLMResponse(
        content="",
        tool_calls=[
            ToolCall(name="read_file", arguments={"path": "a.txt"},
                     id="call_abc"),
        ],
    )
