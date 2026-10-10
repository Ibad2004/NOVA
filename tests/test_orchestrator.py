
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.core.orchestrator import Orchestrator, _is_error_result


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
# Manual test entry point
# --------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
