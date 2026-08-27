from __future__ import annotations

import hashlib
import unicodedata
from enum import StrEnum

import blake3


class HashAlgorithm(StrEnum):
    BLAKE3 = "blake3"
    SHA256 = "sha256"


def normalize_text(text: str) -> str:
    """Canonicalize text before hashing/embedding so formatting-only diffs
    (unicode form, surrounding whitespace) don't register as tampering."""
    return unicodedata.normalize("NFC", text).strip()


def compute_hash(content: str, algorithm: HashAlgorithm = HashAlgorithm.BLAKE3) -> str:
    normalized = normalize_text(content).encode("utf-8")
    if algorithm is HashAlgorithm.BLAKE3:
        return blake3.blake3(normalized).hexdigest()
    if algorithm is HashAlgorithm.SHA256:
        return hashlib.sha256(normalized).hexdigest()
    raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def verify_hash(
    content: str, expected_hash: str, algorithm: HashAlgorithm = HashAlgorithm.BLAKE3
) -> bool:
    return compute_hash(content, algorithm) == expected_hash
