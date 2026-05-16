from pydantic_settings import BaseSettings, SettingsConfigDict

from domain.models.generation import ProviderName


class GenerationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GENERATION_")

    provider: ProviderName = "openai"
    openai_model: str = "gpt-5-nano"
    ollama_model: str = "qwen2.5:3b-instruct"
    temperature: float = 0.2
    max_tokens: int = 512
    max_context_chars: int = 6000
    default_k: int = 8


settings = GenerationSettings()
