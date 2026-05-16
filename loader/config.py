from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOADER_")

    db_url: str = "postgresql+psycopg://postgres:example@localhost:5432/postgres"
    ollama_base_url: str = "http://localhost:11434"
    embed_model_name: str = "nomic-embed-text"
    default_batch_size: int = 1000
    min_year: int = 2023


settings = Settings()
