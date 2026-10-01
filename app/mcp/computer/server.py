import logging
import os

from mcp.server import MCPServer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


mcp = MCPServer("NOVA Computer", log_level="INFO")


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


@mcp.tool()
def get_current_directory() -> str:
    """Return the current working directory."""
    return os.getcwd()


@mcp.tool()
def list_files(path: str = ".") -> str:
    """List files and folders inside the specified directory."""

    try:
        entries = os.listdir(path)

        if not entries:
            return "The directory is empty."

        return "\n".join(entries)

    except FileNotFoundError:
        return f"Directory not found: {path}"

    except NotADirectoryError:
        return f"Not a directory: {path}"

    except PermissionError:
        return f"Permission denied: {path}"


@mcp.tool()
def list_text_files(path: str = ".") -> str:
    """List text files directly inside the specified directory."""

    try:
        matches = []

        for name in os.listdir(path):
            full_path = os.path.join(path, name)

            # Ignore directories.
            if not os.path.isfile(full_path):
                continue

            extension = os.path.splitext(name)[1].lower()

            if extension in TEXT_EXTENSIONS:
                matches.append(full_path)

        if not matches:
            return "No supported text files were found."

        return "\n".join(matches)

    except FileNotFoundError:
        return f"Directory not found: {path}"

    except NotADirectoryError:
        return f"Not a directory: {path}"

    except PermissionError:
        return f"Permission denied: {path}"


@mcp.tool()
def read_file(path: str) -> str:
    """Read the text contents of a file."""

    filename = os.path.basename(path).lower()

    # Protect sensitive configuration files.
    if filename in SENSITIVE_FILES:
        return (
            "Access denied: this file may contain "
            "sensitive information."
        )

    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    except FileNotFoundError:
        return f"File not found: {path}"

    except IsADirectoryError:
        return f"Path is a directory, not a file: {path}"

    except PermissionError:
        return f"Permission denied: {path}"

    except UnicodeDecodeError:
        return (
            f"Unable to read file as UTF-8 text: {path}"
        )


@mcp.tool()
def search_files(path: str, pattern: str) -> str:
    """Search for files and folders matching a pattern in the specified directory."""

    try:
        matches = []

        for name in os.listdir(path):
            if pattern.lower() in name.lower():
                matches.append(
                    os.path.join(path, name)
                )

        if not matches:
            return (
                f"No files or folders matching "
                f"'{pattern}' were found."
            )

        return "\n".join(matches)

    except FileNotFoundError:
        return f"Directory not found: {path}"

    except NotADirectoryError:
        return f"Not a directory: {path}"

    except PermissionError:
        return f"Permission denied: {path}"


if __name__ == "__main__":
    logger.info("NOVA Computer MCP Server is starting...")
    logger.info("Waiting for an MCP client connection...")

    mcp.run()