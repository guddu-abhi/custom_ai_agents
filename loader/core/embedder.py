import numpy as np
import ollama


class EmbeddingService:
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434") -> None:
        self._client = ollama.Client(host=base_url)
        self.model_name = model_name

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dim) float32 array."""
        response = self._client.embed(model=self.model_name, input=texts)
        return np.array(response.embeddings, dtype=np.float32)
