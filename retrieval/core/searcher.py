import asyncio

from domain.models.search import ProductFilters, SearchResult
from otto_lib.embedding import EmbeddingService
from retrieval.db.search_repo import SearchRepository


class SearchService:
    def __init__(self, embedder: EmbeddingService, search_repo: SearchRepository) -> None:
        self._embedder = embedder
        self._search_repo = search_repo

    async def search(
        self, query: str, k: int = 10, filters: ProductFilters | None = None
    ) -> list[SearchResult]:
        # encode_batch is a blocking Ollama HTTP call — keep it off the event loop.
        embedding = (await asyncio.to_thread(self._embedder.encode_batch, [query]))[0]
        return await self._search_repo.search_by_vector(
            embedding=embedding,
            k=k,
            model_name=self._embedder.model_name,
            filters=filters,
        )
