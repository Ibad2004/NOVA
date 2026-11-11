from unittest.mock import MagicMock

import pytest
from google.genai import types

from app.llm.base import LLMResponse, ToolCall
from app.llm.gemini_client import GeminiClient


def make_response(*parts):
    """Build a real GenerateContentResponse with one model candidate."""

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)),
            )
        ]
    )


def make_client(*responses):
    """Build a GeminiClient around a mock SDK client."""

    sdk = MagicMock()
    sdk.models.generate_content.side_effect = list(responses)
    return GeminiClient(client=sdk), sdk.models.generate_content


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiClient()


def test_model_defaults(monkeypatch):
    for name in ("GEMINI_MODEL", "GEMINI_TEMPERATURE", "GEMINI_MAX_TOKENS"):
        monkeypatch.delenv(name, raising=False)

    client = GeminiClient(client=MagicMock())

    assert client.model == "gemini-2.5-flash"
    assert client.temperature == 0.2
    assert client.max_tokens == 2048


def test_text_reply_and_system_instruction():
    client, generate = make_client(
        make_response(types.Part(text="Hello"), types.Part(text=" there!"))
    )

    response = client.chat(
        [
            {"role": "system", "content": "Be helpful."},
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
        think=True,
    )

    assert response == LLMResponse(content="Hello there!", tool_calls=[])

    kwargs = generate.call_args.kwargs
    config = kwargs["config"]
    assert kwargs["model"] == client.model
    assert config.system_instruction == "Be helpful.\n\nBe brief."
    assert config.temperature == client.temperature
    assert config.max_output_tokens == client.max_tokens
    assert config.tools is None
    assert config.automatic_function_calling.disable is True
    assert [c.role for c in kwargs["contents"]] == ["user"]
    assert kwargs["contents"][0].parts[0].text == "Hi"


def test_ask_returns_text():
    client, _ = make_client(make_response(types.Part(text="Islamabad")))

    assert client.ask("Capital of Pakistan?") == "Islamabad"


def test_tool_schema_has_no_dollar_schema():
    client, generate = make_client(make_response(types.Part(text="ok")))
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    client.chat(
        [{"role": "user", "content": "Read a.txt"}],
        tools=[
            {"name": "read_file", "description": "Read a file",
             "parameters": schema},
            {"name": "list_dir", "description": "List a folder",
             "parameters": {"type": "object", "properties": {}}},
        ],
    )

    tools = generate.call_args.kwargs["config"].tools
    assert len(tools) == 1
    declarations = tools[0].function_declarations
    assert [d.name for d in declarations] == ["read_file", "list_dir"]
    assert declarations[0].description == "Read a file"
    assert declarations[0].parameters_json_schema == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    # The caller's schema is not modified.
    assert "$schema" in schema


def test_thought_parts_ignored():
    client, _ = make_client(make_response(
        types.Part(text="Let me think about this...", thought=True),
        types.Part(text="The answer is 4."),
    ))

    response = client.chat([{"role": "user", "content": "2+2?"}])

    assert response == LLMResponse(content="The answer is 4.")


@pytest.mark.parametrize("response", [
    types.GenerateContentResponse(candidates=[]),
    types.GenerateContentResponse(),
    types.GenerateContentResponse(candidates=[types.Candidate()]),
    types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model"))]
    ),
])
def test_empty_candidates_give_empty_response(response):
    client, _ = make_client(response)

    assert client.chat([{"role": "user", "content": "Hi"}]) == LLMResponse()


def test_unknown_role_raises():
    client, generate = make_client()

    with pytest.raises(ValueError, match="narrator"):
        client.chat([{"role": "narrator", "content": "Once upon a time"}])

    generate.assert_not_called()


def test_full_round_trip_with_tool_calls():
    signature = b"opaque-signature-bytes"

    client, generate = make_client(
        make_response(
            types.Part(
                function_call=types.FunctionCall(
                    id="fc_1",
                    name="read_file",
                    args={"path": "a.txt"},
                ),
                thought_signature=signature,
            ),
            types.Part(
                function_call=types.FunctionCall(
                    name="list_dir",
                    args={"path": "."},
                ),
            ),
        ),
        make_response(types.Part(text="a.txt says hi.")),
    )

    messages = [
        {"role": "system", "content": "You are NOVA."},
        {"role": "user", "content": "Read a.txt and list files."},
    ]

    first = client.chat(messages, tools=[
        {"name": "read_file", "description": "", "parameters": {}},
        {"name": "list_dir", "description": "", "parameters": {}},
    ])

    assert first.content == ""
    assert [c.name for c in first.tool_calls] == ["read_file", "list_dir"]
    read_call, list_call = first.tool_calls
    assert read_call.id == "fc_1"
    assert read_call.arguments == {"path": "a.txt"}
    assert read_call.provider_data == {"thought_signature": signature}
    assert list_call.id  # generated because Gemini omitted it
    assert list_call.id != read_call.id
    assert list_call.provider_data == {}

    # Append history exactly like the orchestrator does.
    messages.append({
        "role": "assistant",
        "content": first.content,
        "tool_calls": first.tool_calls,
    })
    for call in first.tool_calls:
        messages.append({
            "role": "tool",
            "content": f"Result from {call.name}:\nok",
            "tool_call_id": call.id,
            "name": call.name,
        })

    second = client.chat(messages)

    assert second == LLMResponse(content="a.txt says hi.")

    contents = generate.call_args.kwargs["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]

    model_parts = contents[1].parts
    assert len(model_parts) == 2
    assert model_parts[0].function_call.id == "fc_1"
    assert model_parts[0].function_call.name == "read_file"
    assert model_parts[0].function_call.args == {"path": "a.txt"}
    assert model_parts[0].thought_signature == signature
    assert model_parts[1].function_call.id == list_call.id
    assert model_parts[1].function_call.name == "list_dir"
    assert model_parts[1].thought_signature is None

    response_parts = contents[2].parts
    assert len(response_parts) == 2
    assert [
        (p.function_response.id, p.function_response.name)
        for p in response_parts
    ] == [("fc_1", "read_file"), (list_call.id, "list_dir")]
    assert response_parts[0].function_response.response == {
        "result": "Result from read_file:\nok"
    }


def test_real_response_object_from_json():
    """Parse a raw API payload into a real GenerateContentResponse."""

    response = types.GenerateContentResponse.model_validate({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"text": "Checking.", "thought": True},
                    {"text": "Reading the file."},
                    {
                        "functionCall": {
                            "id": "fc_9",
                            "name": "read_file",
                            "args": {"path": "notes.md"},
                        },
                        "thoughtSignature": "c2lnbmF0dXJl",
                    },
                ],
            },
            "finishReason": "STOP",
        }],
    })
    client, _ = make_client(response)

    result = client.chat([{"role": "user", "content": "Read notes.md"}])

    assert result.content == "Reading the file."
    assert result.tool_calls == [
        ToolCall(
            name="read_file",
            arguments={"path": "notes.md"},
            id="fc_9",
            provider_data={"thought_signature": b"signature"},
        )
    ]
