import math

from rag_guard.core.embedder import (
    HashingEmbedder,
    cosine_similarity,
    embed_and_normalize,
    normalize_vector,
)


def test_hashing_embedder_is_deterministic() -> None:
    embedder = HashingEmbedder()
    text = "The Eiffel Tower is located in Paris, France."
    assert embedder.embed_query(text) == embedder.embed_query(text)


def test_hashing_embedder_dimensions() -> None:
    embedder = HashingEmbedder(dimensions=64)
    assert len(embedder.embed_query("some chunk of text")) == 64


def test_embed_documents_matches_embed_query() -> None:
    embedder = HashingEmbedder()
    texts = ["first chunk", "second chunk"]
    assert embedder.embed_documents(texts) == [embedder.embed_query(t) for t in texts]


def test_normalize_vector_is_unit_length() -> None:
    normalized = normalize_vector([3.0, 4.0])
    assert math.isclose(math.hypot(*normalized), 1.0, rel_tol=1e-9)


def test_normalize_vector_handles_zero_vector() -> None:
    assert normalize_vector([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_embed_and_normalize_returns_unit_vector() -> None:
    embedder = HashingEmbedder()
    vector = embed_and_normalize(embedder, "some chunk of text")
    norm = math.sqrt(sum(v * v for v in vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-9)


def test_cosine_similarity_identical_vectors_is_one() -> None:
    embedder = HashingEmbedder()
    vector = embed_and_normalize(embedder, "identical content")
    assert math.isclose(cosine_similarity(vector, vector), 1.0, rel_tol=1e-9)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_similar_texts_score_higher_than_unrelated() -> None:
    embedder = HashingEmbedder()
    original = embed_and_normalize(
        embedder, "The Eiffel Tower is located in Paris, France."
    )
    minor_edit = embed_and_normalize(
        embedder, "The Eiffel Tower is located in Paris, Franse."
    )
    unrelated = embed_and_normalize(
        embedder, "Quantum entanglement enables correlated particle measurements."
    )
    assert cosine_similarity(original, minor_edit) > cosine_similarity(original, unrelated)
