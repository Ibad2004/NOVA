from unittest.mock import MagicMock

from app.llm.base import LLMResponse, ToolCall
from app.llm.ollama_client import OllamaClient


def make_client(response_message):
    """Build an OllamaClient around a mock SDK client."""

    sdk = MagicMock()
    sdk.chat.return_value = {"message": response_message}
    return OllamaClient(client=sdk), sdk


def test_text_reply():
    client, sdk = make_client({"role": "assistant", "content": "Hello!"})

    response = client.chat(
        [{"role": "user", "content": "Hi"}],
        think=True,
    )

    assert response == LLMResponse(content="Hello!", tool_calls=[])
    kwargs = sdk.chat.call_args.kwargs
    assert kwargs["model"] == client.model
    assert kwargs["think"] is True
    assert kwargs["options"] == {
        "temperature": client.temperature,
        "num_predict": client.max_tokens,
    }
    assert "tools" not in kwargs


def test_ask_returns_text():
    client, sdk = make_client({"role": "assistant", "content": "Islamabad"})

    assert client.ask("Capital of Pakistan?") == "Islamabad"
    assert sdk.chat.call_args.kwargs["messages"] == [
        {"role": "user", "content": "Capital of Pakistan?"}
    ]


def test_tools_sent_in_ollama_format():
    client, sdk = make_client({"role": "assistant", "content": "ok"})
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

    assert sdk.chat.call_args.kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": schema,
            },
        }
    ]


def test_tool_calls_converted_to_nova_format():
    client, _ = make_client({
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "read_file",
                          "arguments": {"path": "a.txt"}}},
            {"function": {"name": "list_dir", "arguments": {}}},
        ],
    })

    response = client.chat([{"role": "user", "content": "Go"}])

    assert response.content == ""
    assert response.tool_calls == [
        ToolCall(name="read_file", arguments={"path": "a.txt"},
                 id="call_0"),
        ToolCall(name="list_dir", arguments={}, id="call_1"),
    ]


def test_history_converted_to_ollama_format():
    client, sdk = make_client({"role": "assistant", "content": "Done"})

    client.chat([
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Read a.txt"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                ToolCall(name="read_file", arguments={"path": "a.txt"},
                         id="call_0"),
            ],
        },
        {
            "role": "tool",
            "content": "FILE_READ_SUCCESS",
            "tool_call_id": "call_0",
            "name": "read_file",
        },
    ])

    assert sdk.chat.call_args.kwargs["messages"] == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Read a.txt"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file",
                              "arguments": {"path": "a.txt"}}},
            ],
        },
        {
            "role": "tool",
            "content": "FILE_READ_SUCCESS",
            "tool_name": "read_file",
        },
    ]


def test_sdk_response_objects_are_supported():
    """The real SDK returns ChatResponse objects, not dicts."""

    from ollama import ChatResponse, Message

    sdk = MagicMock()
    sdk.chat.return_value = ChatResponse(
        message=Message(
            role="assistant",
            content="",
            tool_calls=[
                Message.ToolCall(function=Message.ToolCall.Function(
                    name="read_file", arguments={"path": "a.txt"},
                )),
            ],
        )
    )

    response = OllamaClient(client=sdk).chat(
        [{"role": "user", "content": "Go"}]
    )

    assert response.tool_calls == [
        ToolCall(name="read_file", arguments={"path": "a.txt"},
                 id="call_0"),
    ]
