
import logging
import os

from mcp.server import MCPServer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = MCPServer("NOVA Computer", log_level="INFO")


# Restrict file operations to the NOVA project directory.
WORKSPACE_ROOT = os.path.realpath(os.getcwd())


# Files that may contain secrets or sensitive configuration.
SENSITIVE_FILES = {
    ".env",
    ".env.local",
    ".env.production",
}


# File extensions that are normally text-based.
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".sql",
    ".ini",
    ".toml",
}


def validate_path(path: str) -> str:
    """Resolve a path and ensure it stays inside the NOVA workspace."""

    resolved_path = os.path.realpath(
        os.path.abspath(
            os.path.join(WORKSPACE_ROOT, path)
        )
    )

    try:
        inside_workspace = (
            os.path.commonpath(
                [WORKSPACE_ROOT, resolved_path]
            ) == WORKSPACE_ROOT
        )
    except ValueError:
        inside_workspace = False

    if not inside_workspace:
        raise ValueError(
            "Access denied: path is outside the NOVA workspace."
        )

    return resolved_path


@mcp.tool()
def get_current_directory() -> str:
    """Return the current working directory."""

    return WORKSPACE_ROOT


@mcp.tool()
def list_files(path: str = ".") -> str:
    """List files and folders inside the NOVA workspace."""

    try:
        safe_path = validate_path(path)

        if not os.path.isdir(safe_path):
            return f"Not a directory: {path}"

        entries = os.listdir(safe_path)

        if not entries:
            return "The directory is empty."

        return "\n".join(entries)

    except ValueError as error:
        return str(error)

    except FileNotFoundError:
        return f"Directory not found: {path}"

    except PermissionError:
        return f"Permission denied: {path}"


@mcp.tool()
def list_text_files(path: str = ".") -> str:
    """List text files directly inside a workspace directory."""

    try:
        safe_path = validate_path(path)

        if not os.path.isdir(safe_path):
            return f"Not a directory: {path}"

        matches = []

        for name in os.listdir(safe_path):
            full_path = validate_path(
                os.path.join(safe_path, name)
            )

            if not os.path.isfile(full_path):
                continue

            extension = os.path.splitext(name)[1].lower()

            if extension in TEXT_EXTENSIONS:
                matches.append(full_path)

        if not matches:
            return "No supported text files were found."

        return "\n".join(matches)

    except ValueError as error:
        return str(error)

    except FileNotFoundError:
        return f"Directory not found: {path}"

    except PermissionError:
        return f"Permission denied: {path}"



@mcp.tool()
def read_file(path: str) -> str:
    """Read a text file and clearly report its actual contents."""

    try:
        safe_path = validate_path(path)
        filename = os.path.basename(safe_path).lower()

        if filename in SENSITIVE_FILES:
            return "ERROR: Access denied. Sensitive file."

        if not os.path.isfile(safe_path):
            return f"ERROR: File not found: {path}"

        with open(safe_path, "r", encoding="utf-8") as file:
            content = file.read()

        return (
            f"FILE_READ_SUCCESS\n"
            f"Path: {safe_path}\n"
            f"Content length: {len(content)} characters\n"
            f"--- BEGIN FILE CONTENT ---\n"
            f"{content}\n"
            f"--- END FILE CONTENT ---"
        )

    except ValueError as error:
        return f"ERROR: {error}"
    except FileNotFoundError:
        return f"ERROR: File not found: {path}"
    except PermissionError:
        return f"ERROR: Permission denied: {path}"
    except UnicodeDecodeError:
        return f"ERROR: Unable to read UTF-8 text: {path}"




@mcp.tool()
def validate_python_file(path: str) -> str:
    """Check Python syntax without executing the file."""

    try:
        safe_path = validate_path(path)
        filename = os.path.basename(safe_path).lower()

        if filename in SENSITIVE_FILES:
            return "ERROR: Access denied. Sensitive file."

        if not safe_path.lower().endswith(".py"):
            return "ERROR: Only Python (.py) files can be validated."

        if not os.path.isfile(safe_path):
            return f"ERROR: Python file not found: {path}"

        with open(safe_path, "r", encoding="utf-8") as file:
            source = file.read()

        compile(source, safe_path, "exec")

        return (
            f"PYTHON_SYNTAX_VALID\n"
            f"File: {safe_path}\n"
            "Python syntax is valid. Code was not executed."
        )

    except SyntaxError as error:
        return (
            f"PYTHON_SYNTAX_ERROR\n"
            f"Line: {error.lineno}\n"
            f"Error: {error.msg}"
        )

    except ValueError as error:
        return f"ERROR: {error}"

    except FileNotFoundError:
        return f"ERROR: Python file not found: {path}"

    except PermissionError:
        return f"ERROR: Permission denied: {path}"

    except UnicodeDecodeError:
        return f"ERROR: Unable to read UTF-8 Python file: {path}"







@mcp.tool()
def create_file(path: str, content: str) -> str:
    """Create a new text file inside the NOVA workspace without overwriting existing files."""

    try:
        safe_path = validate_path(path)

        filename = os.path.basename(safe_path).lower()

        if filename in SENSITIVE_FILES:
            return "Access denied: cannot create a sensitive configuration file."

        if os.path.exists(safe_path):
            return (
                f"Creation refused: a file or directory already exists "
                f"at {path}. Existing content was not changed."
            )

        parent_directory = os.path.dirname(safe_path)

        if not os.path.isdir(parent_directory):
            return (
                f"Creation failed: parent directory does not exist: "
                f"{parent_directory}"
            )

        with open(safe_path, "x", encoding="utf-8") as file:
            file.write(content)

        if not os.path.isfile(safe_path):
            return "Creation could not be verified."

        return f"File created successfully: {safe_path}"

    except ValueError as error:
        return str(error)

    except FileExistsError:
        return (
            f"Creation refused: a file already exists at {path}. "
            "Existing content was not changed."
        )

    except PermissionError:
        return f"Permission denied: {path}"

    except OSError as error:
        return f"File creation failed: {error}"




@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Append text to an existing file without replacing its content."""

    try:
        safe_path = validate_path(path)
        filename = os.path.basename(safe_path).lower()

        if filename in SENSITIVE_FILES:
            return "Access denied: sensitive file."

        if not os.path.isfile(safe_path):
            return f"File not found: {path}"

        with open(safe_path, "a", encoding="utf-8") as file:
            file.write(content)

        return f"Content appended successfully: {safe_path}"

    except ValueError as error:
        return str(error)
    except PermissionError:
        return f"Permission denied: {path}"
    except OSError as error:
        return f"Write failed: {error}"







@mcp.tool()
def search_files(path: str, pattern: str) -> str:
    """Search filenames inside a NOVA workspace directory."""

    try:
        safe_path = validate_path(path)

        if not os.path.isdir(safe_path):
            return f"Not a directory: {path}"

        matches = []

        for name in os.listdir(safe_path):
            full_path = validate_path(
                os.path.join(safe_path, name)
            )

            if pattern.lower() in name.lower():
                matches.append(full_path)

        if not matches:
            return (
                f"No files or folders matching "
                f"'{pattern}' were found."
            )

        return "\n".join(matches)

    except ValueError as error:
        return str(error)

    except FileNotFoundError:
        return f"Directory not found: {path}"

    except PermissionError:
        return f"Permission denied: {path}"


if __name__ == "__main__":
    logger.info("NOVA Computer MCP Server is starting...")
    logger.info("Workspace: %s", WORKSPACE_ROOT)
    logger.info("Waiting for an MCP client connection...")

    mcp.run()
