"""Tests for the RAG module."""

from sherlock.rag import chunk_text, MockEmbedder, VectorStore, _cosine


class TestChunking:
    def test_empty_text(self):
        assert chunk_text("") == []

    def test_short_text_single_chunk(self):
        assert chunk_text("hello world") == ["hello world"]

    def test_long_text_multiple_chunks(self):
        text = " ".join(str(i) for i in range(700))
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        assert len(chunks) >= 2

    def test_overlap_applied(self):
        text = " ".join(str(i) for i in range(400))
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        # Second chunk should start before the first chunk ends (overlap)
        first_words = chunks[0].split()
        second_words = chunks[1].split()
        assert second_words[0] in first_words


class TestMockEmbedder:
    def test_dimension(self):
        emb = MockEmbedder()
        vecs = emb.embed(["hello world"])
        assert len(vecs) == 1
        assert len(vecs[0]) == emb.dim

    def test_deterministic(self):
        emb = MockEmbedder()
        a = emb.embed(["redis connection timeout"])[0]
        b = emb.embed(["redis connection timeout"])[0]
        assert a == b

    def test_normalized(self):
        emb = MockEmbedder()
        vec = emb.embed(["some text here"])[0]
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_empty_list(self):
        assert MockEmbedder().embed([]) == []


class TestVectorStore:
    def test_add_and_size(self):
        store = VectorStore(MockEmbedder())
        n = store.add(["doc one", "doc two"])
        assert n == 2
        assert store.size == 2

    def test_search_empty_store(self):
        store = VectorStore(MockEmbedder())
        assert store.search("anything") == []

    def test_search_returns_relevant(self):
        store = VectorStore(MockEmbedder())
        store.add([
            "redis connection failures should fall back to postgres",
            "the bakery sells fresh bread every morning",
            "kubernetes pods restart on health check failure",
        ])
        results = store.search("redis fallback postgres", k=1)
        assert len(results) == 1
        # The redis doc should rank first for a redis query
        assert "redis" in results[0][0].lower()

    def test_search_respects_k(self):
        store = VectorStore(MockEmbedder())
        store.add(["a b c", "d e f", "g h i", "j k l"])
        assert len(store.search("a b c", k=2)) == 2


class TestCosine:
    def test_identical_vectors(self):
        assert abs(_cosine([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        assert abs(_cosine([1, 0], [0, 1])) < 1e-9
