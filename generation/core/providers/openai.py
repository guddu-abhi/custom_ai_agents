from openai import OpenAI

from domain.models.generation import Usage
from generation.core.providers.base import GenerationParams, LLMResponse


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        self._client = OpenAI()

    def complete(self, system: str, user: str, params: GenerationParams) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=params.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            top_p=params.top_p,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        u = resp.usage
        usage = Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            total_tokens=getattr(u, "total_tokens", 0) or 0,
        )
        return LLMResponse(text=text, usage=usage)
