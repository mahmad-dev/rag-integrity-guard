import pytest
from fastapi.testclient import TestClient

import index as index_module
import state as state_module
from index import app
from rag_guard.core.embedder import embed_and_normalize

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_state():
    state_module._state = None
    yield
    state_module._state = None


def _ingest_sample() -> dict:
    response = client.post(
        "/api/ingest",
        json={
            "document_id": "doc-1",
            "chunks": [
                {"text": "The Eiffel Tower is located in Paris, France."},
                {"text": "Mount Everest is the tallest mountain above sea level."},
            ],
        },
    )
    assert response.status_code == 200
    return response.json()


def test_ingest_returns_chunk_ids_and_hashes() -> None:
    body = _ingest_sample()
    assert body["document_id"] == "doc-1"
    assert len(body["ingested"]) == 2
    assert body["ingested"][0]["chunk_id"] == "doc-1:0"
    assert len(body["ingested"][0]["content_hash"]) > 0


def test_query_returns_valid_unmodified_chunk() -> None:
    _ingest_sample()

    response = client.post("/api/query", json={"query": "Where is the Eiffel Tower?", "k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["blocked"] is False
    assert len(body["results"]) == 1
    assert body["results"][0]["chunk_id"] == "doc-1:0"
    assert body["results"][0]["integrity_status"] == "valid"


def test_attack_tampers_the_live_chunk_without_touching_the_fingerprint() -> None:
    _ingest_sample()

    response = client.post(
        "/api/attack", json={"chunk_id": "doc-1:0", "attack_type": "exact_mutation"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_id"] == "doc-1:0"
    assert body["original_excerpt"] != body["tampered_excerpt"]
    assert body["resulting_status"] != "valid"


def test_filter_mode_drops_tampered_chunk_without_blocking() -> None:
    _ingest_sample()
    client.post("/api/attack", json={"chunk_id": "doc-1:0", "attack_type": "exact_mutation"})

    response = client.post(
        "/api/query",
        json={"query": "Where is the Eiffel Tower?", "k": 2, "strictness": "filter"},
    )

    body = response.json()
    assert body["blocked"] is False
    chunk_ids = [r["chunk_id"] for r in body["results"]]
    assert "doc-1:0" not in chunk_ids


def test_strict_mode_blocks_the_whole_query_on_tampering() -> None:
    _ingest_sample()
    client.post("/api/attack", json={"chunk_id": "doc-1:0", "attack_type": "payload_injection"})

    response = client.post(
        "/api/query",
        json={"query": "Eiffel Tower Paris", "k": 1, "strictness": "strict"},
    )

    body = response.json()
    assert body["blocked"] is True
    assert body["results"] == []
    assert len(body["violations"]) == 1
    assert "doc-1:0" in body["violations"][0]


def test_semantic_drift_attack_replaces_chunk_with_pool_member() -> None:
    _ingest_sample()

    response = client.post(
        "/api/attack", json={"chunk_id": "doc-1:0", "attack_type": "semantic_drift"}
    )

    body = response.json()
    assert body["tampered_excerpt"].startswith("Mount Everest")


def test_attack_on_unknown_chunk_returns_404() -> None:
    response = client.post(
        "/api/attack", json={"chunk_id": "does-not-exist", "attack_type": "exact_mutation"}
    )
    assert response.status_code == 404


def test_benchmark_returns_precomputed_report() -> None:
    response = client.get("/api/benchmark")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == "pietrolesci/nli_fever"
    assert body["overall_true_positive_rate"] == 1.0
    assert body["overall_false_positive_rate"] == 0.0
    assert len(body["per_attack_type"]) == 4


def test_benchmark_503s_when_no_report_is_bundled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(index_module, "_BENCHMARK_REPORT_PATH", tmp_path / "missing.json")

    response = client.get("/api/benchmark")

    assert response.status_code == 503


def test_attack_on_a_chunk_with_no_fingerprint_reports_missing_fingerprint() -> None:
    state = state_module.get_state()
    text = "A chunk that made it into the vector store without ever being fingerprinted."
    state.collection.add(
        ids=["orphan:0"],
        documents=[text],
        embeddings=[embed_and_normalize(state.embedder, text)],
        metadatas=[{"chunk_id": "orphan:0"}],
    )

    response = client.post(
        "/api/attack", json={"chunk_id": "orphan:0", "attack_type": "exact_mutation"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["resulting_status"] == "missing_fingerprint"
    assert body["resulting_similarity"] is None
