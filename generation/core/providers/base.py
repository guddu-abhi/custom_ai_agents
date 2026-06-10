from dataclasses import dataclass
from typing import Protocol

from domain.models.generation import Usage


@dataclass(frozen=True)
class GenerationParams:
    model: str
    temperature: float = 0.2
    max_tokens: int = 512
    top_p: float = 1.0


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str, params: GenerationParams) -> LLMResponse: ...
