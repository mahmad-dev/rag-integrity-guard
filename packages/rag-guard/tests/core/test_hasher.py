import unicodedata

import pytest

from rag_guard.core.hasher import HashAlgorithm, compute_hash, normalize_text, verify_hash


@pytest.mark.parametrize("algorithm", [HashAlgorithm.BLAKE3, HashAlgorithm.SHA256])
def test_compute_hash_is_deterministic(algorithm: HashAlgorithm) -> None:
    content = "The Eiffel Tower is located in Paris, France."
    assert compute_hash(content, algorithm) == compute_hash(content, algorithm)


@pytest.mark.parametrize("algorithm", [HashAlgorithm.BLAKE3, HashAlgorithm.SHA256])
def test_compute_hash_changes_on_tamper(algorithm: HashAlgorithm) -> None:
    original = "The Eiffel Tower is located in Paris, France."
    tampered = "The Eiffel Tower is located in London, France."
    assert compute_hash(original, algorithm) != compute_hash(tampered, algorithm)


def test_blake3_and_sha256_diverge_for_same_content() -> None:
    content = "cross-algorithm digests should not collide"
    assert compute_hash(content, HashAlgorithm.BLAKE3) != compute_hash(
        content, HashAlgorithm.SHA256
    )


def test_verify_hash_roundtrip() -> None:
    content = "verify me"
    digest = compute_hash(content)
    assert verify_hash(content, digest)
    assert not verify_hash("verify me!", digest)


def test_normalize_text_strips_whitespace() -> None:
    assert normalize_text("  hello world  \n") == "hello world"


def test_normalize_text_unifies_unicode_forms() -> None:
    # NFC-composed vs NFD-decomposed encodings of the same visible glyph must
    # hash identically, or cosmetic re-encoding would register as tampering.
    base = "café"  # "cafe" + combining acute accent (NFD-ish)
    composed = unicodedata.normalize("NFC", base)
    decomposed = unicodedata.normalize("NFD", base)
    assert composed != decomposed
    assert compute_hash(composed) == compute_hash(decomposed)


def test_whitespace_only_diffs_do_not_change_hash() -> None:
    assert compute_hash("hello world") == compute_hash("  hello world  ")
