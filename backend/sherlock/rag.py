"""
Retrieval-Augmented Generation (RAG) for Sherlock.

Provides document chunking, embeddings, and a vector store with cosine-similarity
search. Embeddings come from an LLM provider's embedding API when a key is
available; otherwise a deterministic hashing embedder keeps everything runnable
and testable offline. The VectorStore is intentionally simple (numpy cosine
similarity) and pluggable — swap in FAISS, Chroma, or pgvector for scale.
"""

import os
import hashlib
import math
from abc import ABC, abstractmethod


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-based chunks for embedding."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


class BaseEmbedder(ABC):
    dim: int = 256
    mode: str = "mock"

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbedder(BaseEmbedder):
    """Embeddings via OpenAI's embedding API."""

    mode = "live"
    dim = 1536

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class MockEmbedder(BaseEmbedder):
    """
    Deterministic hashing embedder. Maps tokens into a fixed-dimension vector
    via hashing, then L2-normalizes. Not semantically perfect, but stable and
    dependency-free — good enough to exercise retrieval in tests and demos.
    """

    mode = "mock"
    dim = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class VectorStore:
    """In-memory vector store with cosine-similarity search."""

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self._docs: list[str] = []
        self._vecs: list[list[float]] = []

    def add(self, texts: list[str]) -> int:
        if not texts:
            return 0
        vecs = self.embedder.embed(texts)
        self._docs.extend(texts)
        self._vecs.extend(vecs)
        return len(texts)

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        if not self._docs:
            return []
        qvec = self.embedder.embed([query])[0]
        scored = [(doc, _cosine(qvec, vec)) for doc, vec in zip(self._docs, self._vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    @property
    def size(self) -> int:
        return len(self._docs)


def get_embedder() -> BaseEmbedder:
    """Factory: API embedder if a key exists, else deterministic mock."""
    openai_key = os.getenv("OPENAI_API_KEY")
    provider = os.getenv("SHERLOCK_EMBEDDER", "auto").lower()
    if provider in ("openai", "auto") and openai_key:
        return OpenAIEmbedder(api_key=openai_key)
    return MockEmbedder()
