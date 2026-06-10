from typing import Any

from openai import OpenAI

from domain.models.generation import Usage
from generation.core.providers.base import GenerationParams, LLMResponse


def _is_reasoning_model(model: str) -> bool:
    """GPT-5 / o-series reasoning models reject custom sampling params and
    require ``max_completion_tokens`` instead of ``max_tokens``."""
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        self._client = OpenAI()

    def complete(self, system: str, user: str, params: GenerationParams) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": params.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # max_tokens is deprecated and rejected by GPT-5/o-series; the
            # replacement is accepted by current models across the board.
            "max_completion_tokens": params.max_tokens,
        }
        # Reasoning models only allow default temperature/top_p (=1); omit them.
        if not _is_reasoning_model(params.model):
            kwargs["temperature"] = params.temperature
            kwargs["top_p"] = params.top_p

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        u = resp.usage
        usage = Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            total_tokens=getattr(u, "total_tokens", 0) or 0,
        )
        return LLMResponse(text=text, usage=usage)
