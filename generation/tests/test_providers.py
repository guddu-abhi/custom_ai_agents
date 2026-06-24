import os

import pytest

from loader.config import settings as loader_settings
from otto_lib.llm.base import GenerationParams
from otto_lib.llm.ollama import OllamaProvider
from otto_lib.llm.openai import OpenAIProvider


@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_provider_returns_text() -> None:
    provider = OpenAIProvider()
    out = provider.complete(
        "Reply with the single word: ok.",
        "Hello.",
        GenerationParams(model=os.getenv("GENERATION_TEST_OPENAI_MODEL", "gpt-5-nano"),
                         temperature=0.0, max_tokens=4),
    )
    assert out.text.strip() != ""
    assert out.usage.total_tokens > 0


@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("RUN_OLLAMA_TESTS"), reason="set RUN_OLLAMA_TESTS=1 to run")
def test_ollama_provider_returns_text() -> None:
    provider = OllamaProvider(base_url=loader_settings.ollama_base_url)
    out = provider.complete(
        "Reply with the single word: ok.",
        "Hello.",
        GenerationParams(
            model=os.getenv("GENERATION_TEST_OLLAMA_MODEL", "qwen2.5:3b-instruct"),
            temperature=0.0,
            max_tokens=8,
        ),
    )
    assert out.text.strip() != ""
