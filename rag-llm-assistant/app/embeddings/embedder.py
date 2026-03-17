"""Embedder module — wraps sentence-transformers for text embedding."""
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Manages embedding model lifecycle and caching."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._cache = {}

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts):
        """Embed a list of texts or a single text string."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return np.array(embeddings, dtype="float32")

    def embed_cached(self, texts):
        """Embed with caching for repeated texts."""
        if isinstance(texts, str):
            texts = [texts]
        results = []
        to_embed = []
        indices = []
        for i, text in enumerate(texts):
            if text in self._cache:
                results.append((i, self._cache[text]))
            else:
                to_embed.append(text)
                indices.append(i)
        if to_embed:
            new_embeddings = self.model.encode(to_embed, convert_to_numpy=True)
            for idx, text, emb in zip(indices, to_embed, new_embeddings):
                self._cache[text] = emb
                results.append((idx, emb))
        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results], dtype="float32")

    def clear_cache(self):
        self._cache.clear()