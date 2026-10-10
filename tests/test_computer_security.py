from app.mcp.computer.server import write_file
import app.mcp.computer.server as computer_server
from app.mcp.computer.server import validate_python_file
from app.mcp.computer.server import create_file
from app.mcp.computer.server import (
    WORKSPACE_ROOT,
    validate_path,
    list_files,
    read_file,
)


def test_workspace_root_is_allowed():
    assert validate_path(".") == WORKSPACE_ROOT


def test_path_inside_workspace_is_allowed():
    path = validate_path("app")
    assert path.startswith(WORKSPACE_ROOT)


def test_parent_directory_is_rejected():
    try:
        validate_path("..\\..")
        assert False, "Outside path should be rejected"
    except ValueError as error:
        assert "outside the NOVA workspace" in str(error)


def test_env_file_is_blocked():
    result = read_file(".env")
    assert "Access denied" in result


def test_outside_directory_listing_is_blocked():
    result = list_files("..\\..")
    assert "Access denied" in result



def test_existing_file_cannot_be_overwritten(tmp_path):
    test_file = tmp_path / "agent_test.txt"
    original_content = "Original protected content."

    test_file.write_text(original_content, encoding="utf-8")

    # Convert the temporary file into a path relative to the workspace.
    workspace_test_file = (
        test_file.relative_to(WORKSPACE_ROOT)
        if test_file.is_relative_to(WORKSPACE_ROOT)
        else None
    )

    if workspace_test_file is None:
        # Use the existing workspace fixture instead.
        test_path = "data/workspace/agent_test.txt"

        with open(test_path, "r", encoding="utf-8") as file:
            original_content = file.read()

        result = create_file(
            test_path,
            "This should not replace the original content."
        )

        assert "Creation refused" in result

        with open(test_path, "r", encoding="utf-8") as file:
            assert file.read() == original_content
    else:
        result = create_file(
            str(workspace_test_file),
            "This should not replace the original content."
        )
        assert "Creation refused" in result
        assert test_file.read_text(encoding="utf-8") == original_content



def test_file_creation_outside_workspace_is_blocked():
    result = create_file(
        "..\\..\\outside_test.txt",
        "This should not be created."
    )
    assert "Access denied" in result




def test_write_file_appends_without_overwriting(tmp_path):
    test_path = "data/workspace/write_test.txt"

    with open(test_path, "w", encoding="utf-8") as file:
        file.write("Original text.")

    result = write_file(test_path, " Added text.")

    assert "Content appended successfully" in result

    with open(test_path, "r", encoding="utf-8") as file:
        assert file.read() == "Original text. Added text."

def test_valid_python_syntax(tmp_path, monkeypatch):
    monkeypatch.setattr(
        computer_server, "WORKSPACE_ROOT", str(tmp_path)
    )
    (tmp_path / "valid.py").write_text(
        "print('Hello')\n", encoding="utf-8"
    )

    result = validate_python_file("valid.py")

    assert "PYTHON_SYNTAX_VALID" in result
    assert "not executed" in result


def test_invalid_python_syntax(tmp_path, monkeypatch):
    monkeypatch.setattr(
        computer_server, "WORKSPACE_ROOT", str(tmp_path)
    )
    (tmp_path / "invalid.py").write_text(
        "if True print('Hello')\n", encoding="utf-8"
    )

    result = validate_python_file("invalid.py")

    assert "PYTHON_SYNTAX_ERROR" in result
    assert "Line: 1" in result


def test_non_python_file_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        computer_server, "WORKSPACE_ROOT", str(tmp_path)
    )
    (tmp_path / "notes.txt").write_text(
        "Some text", encoding="utf-8"
    )

    result = validate_python_file("notes.txt")

    assert "Only Python (.py) files" in result


def test_missing_python_file_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        computer_server, "WORKSPACE_ROOT", str(tmp_path)
    )

    result = validate_python_file("missing.py")

    assert "not found" in result.lower()


def test_python_file_outside_workspace_is_rejected():
    result = validate_python_file("..\\..\\outside.py")

    assert "outside the NOVA workspace" in result
