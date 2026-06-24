from pydantic_settings import SettingsConfigDict

from otto_lib.config import BaseAppSettings


class OttoSettings(BaseAppSettings):
    model_config = SettingsConfigDict(env_prefix="OTTO_")

    planner_model: str = "gpt-4o-mini"
    answerer_model: str = "gpt-4o"
    # Single-agent path (single_agent.py): one model owns planning + answering
    # and calls the retriever as a function_tool.
    single_agent_model: str = "gpt-4o"
    retrieve_k: int = 20
    final_k: int = 8
    max_context_chars: int = 6000
    env: str = "local"
    # Conversational endpoint: how many prior turns (user+assistant pairs) to
    # feed back into the planner / answerer as context.
    history_max_turns: int = 6


settings = OttoSettings()
