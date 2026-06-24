from pydantic_settings import SettingsConfigDict

from otto_lib.config import BaseAppSettings


class RetrievalSettings(BaseAppSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    default_k: int = 10
    similarity_threshold: float = 0.6


settings = RetrievalSettings()
