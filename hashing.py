"""
hashing.py

Utility functions for deterministic SHA-256 hashing.
"""

from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    """
    Stable SHA-256 hash for any text.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()