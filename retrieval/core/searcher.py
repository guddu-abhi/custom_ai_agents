from domain.models.search_plan import ProductFilters
from loader.core.embedder import EmbeddingService
from retrieval.db.search_repo import SearchRepository, SearchResult


class SearchService:
    def __init__(self, embedder: EmbeddingService, search_repo: SearchRepository) -> None:
        self._embedder = embedder
        self._search_repo = search_repo

    def search(
        self, query: str, k: int = 10, filters: ProductFilters | None = None
    ) -> list[SearchResult]:
        embedding = self._embedder.encode_batch([query])[0]
        return self._search_repo.search_by_vector(
            embedding=embedding,
            k=k,
            model_name=self._embedder.model_name,
            filters=filters,
        )
