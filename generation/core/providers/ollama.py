import ollama

from domain.models.generation import Usage
from generation.core.providers.base import GenerationParams, LLMResponse


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str) -> None:
        self._client = ollama.Client(host=base_url)

    def complete(self, system: str, user: str, params: GenerationParams) -> LLMResponse:
        resp = self._client.chat(
            model=params.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={
                "temperature": params.temperature,
                "num_predict": params.max_tokens,
                "top_p": params.top_p,
            },
        )
        text = resp.get("message", {}).get("content", "") if isinstance(resp, dict) else (
            getattr(getattr(resp, "message", None), "content", "") or ""
        )
        prompt_tokens = _get(resp, "prompt_eval_count") or 0
        completion_tokens = _get(resp, "eval_count") or 0
        usage = Usage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(prompt_tokens) + int(completion_tokens),
        )
        return LLMResponse(text=text, usage=usage)


def _get(resp: object, key: str) -> int | None:
    if isinstance(resp, dict):
        v = resp.get(key)
    else:
        v = getattr(resp, key, None)
    return v if isinstance(v, int) else None
