from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    default_k: int = 10
    similarity_threshold: float = 0.6


settings = RetrievalSettings()
