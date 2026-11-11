import pytest

from app.llm.factory import get_llm_client
# Import the client modules up front so their load_dotenv() calls run
# before monkeypatch adjusts the environment in each test.
from app.llm.gemini_client import GeminiClient
from app.llm.groq_client import GroqClient
from app.llm.ollama_client import OllamaClient


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("LLM_PROVIDER", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_default_is_ollama():
    assert isinstance(get_llm_client(), OllamaClient)


def test_env_selects_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-key")

    assert isinstance(get_llm_client(), GeminiClient)


def test_env_selects_groq(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "dummy-groq-key")

    assert isinstance(get_llm_client(), GroqClient)


@pytest.mark.parametrize("value", ["  GROQ ", "Groq", "groq\n"])
def test_env_is_case_and_whitespace_insensitive(monkeypatch, value):
    monkeypatch.setenv("LLM_PROVIDER", value)
    monkeypatch.setenv("GROQ_API_KEY", "dummy-groq-key")

    assert isinstance(get_llm_client(), GroqClient)


def test_argument_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    assert isinstance(get_llm_client(" Ollama "), OllamaClient)


@pytest.mark.parametrize("provider, key", [
    ("gemini", "GEMINI_API_KEY"),
    ("groq", "GROQ_API_KEY"),
])
def test_missing_key_error_surfaces(monkeypatch, provider, key):
    monkeypatch.setenv("LLM_PROVIDER", provider)

    with pytest.raises(RuntimeError, match=key):
        get_llm_client()


@pytest.mark.parametrize("source", ["env", "argument"])
def test_unknown_provider_raises(monkeypatch, source):
    if source == "env":
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        call = get_llm_client
    else:
        def call():
            return get_llm_client("openai")

    with pytest.raises(ValueError) as error:
        call()

    message = str(error.value)
    assert "'openai'" in message
    for name in ("ollama", "gemini", "groq"):
        assert name in message
