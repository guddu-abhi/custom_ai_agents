from generation.config import settings
from generation.core.providers.base import LLMProvider
from generation.core.providers.ollama import OllamaProvider
from generation.core.providers.openai import OpenAIProvider
from domain.models.generation import ProviderName

from loader.config import settings as loader_settings


def get_provider(name: ProviderName) -> LLMProvider:
    if name == "openai":
        return OpenAIProvider()
    if name == "ollama":
        return OllamaProvider(base_url=loader_settings.ollama_base_url)
    raise ValueError(f"Unknown provider: {name!r}")


__all__ = ["LLMProvider", "OpenAIProvider", "OllamaProvider", "get_provider", "settings"]
