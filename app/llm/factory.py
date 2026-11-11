import os

from dotenv import load_dotenv

from app.llm.base import BaseLLMClient

load_dotenv()


SUPPORTED_PROVIDERS = ("ollama", "gemini", "groq")


def get_llm_client(provider: str | None = None) -> BaseLLMClient:
    """Create the LLM client for `provider` (or LLM_PROVIDER).

    Client modules are imported lazily, so only the chosen provider's
    SDK needs to be installed.
    """

    name = (
        provider
        if provider is not None
        else os.getenv("LLM_PROVIDER", "ollama")
    ).strip().lower()

    if name == "ollama":
        from app.llm.ollama_client import OllamaClient
        return OllamaClient()

    if name == "gemini":
        from app.llm.gemini_client import GeminiClient
        return GeminiClient()

    if name == "groq":
        from app.llm.groq_client import GroqClient
        return GroqClient()

    raise ValueError(
        f"Unknown LLM provider: {name!r}. "
        f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
    )
