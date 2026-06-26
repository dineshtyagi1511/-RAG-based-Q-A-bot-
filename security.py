"""
security.py — Three-tier security layer for the RAG Q&A Bot.

Tier 1 — Input Sanitization  : regex-based, zero-dependency, always active.
Tier 2 — Rate Limiting        : Redis sliding-window counter per session.
Tier 3 — LLM Guard Scanners   : ML-based prompt-injection & toxicity checks
                                 (optional; activated via ENABLE_LLM_GUARD=true).
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Tuple

import redis as redis_lib

from config import (
    ENABLE_LLM_GUARD,
    MAX_QUERY_LENGTH,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW,
    REDIS_URL,
)

logger = logging.getLogger(__name__)

# ── Redis connection (shared with cache) ─────────────────────────────────────
try:
    _redis = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
    _redis.ping()
    REDIS_OK = True
    logger.info("✅ Security: Redis connected")
except Exception as exc:
    REDIS_OK = False
    logger.warning(f"⚠️  Security: Redis unavailable ({exc}). Rate limiting disabled.")

# ── LLM Guard scanners (lazy-loaded to avoid slow import at startup) ──────────
_input_scanners: list | None = None
_output_scanners: list | None = None


def _load_input_scanners() -> list:
    global _input_scanners
    if _input_scanners is not None:
        return _input_scanners

    if not ENABLE_LLM_GUARD:
        _input_scanners = []
        return _input_scanners

    try:
        from llm_guard import scan_prompt  # noqa: F401  (import test)
        from llm_guard.input_scanners import PromptInjection, Toxicity, TokenLimit
        from llm_guard.input_scanners.prompt_injection import MatchType

        _input_scanners = [
            TokenLimit(limit=512),                          # fast, first gate
            PromptInjection(threshold=0.75, match_type=MatchType.FULL),
            Toxicity(threshold=0.70),
        ]
        logger.info("✅ LLM Guard input scanners loaded")
    except ImportError:
        logger.warning("⚠️  llm-guard not installed — ML scanning skipped")
        _input_scanners = []
    except Exception as exc:
        logger.error(f"LLM Guard init error: {exc}")
        _input_scanners = []

    return _input_scanners


def _load_output_scanners() -> list:
    global _output_scanners
    if _output_scanners is not None:
        return _output_scanners

    if not ENABLE_LLM_GUARD:
        _output_scanners = []
        return _output_scanners

    try:
        from llm_guard.output_scanners import Relevance

        _output_scanners = [
            Relevance(threshold=0.10),   # catch completely off-topic answers
        ]
        logger.info("✅ LLM Guard output scanners loaded")
    except ImportError:
        _output_scanners = []
    except Exception as exc:
        logger.error(f"LLM Guard output init error: {exc}")
        _output_scanners = []

    return _output_scanners


# ── Tier 1 — Input Sanitization ──────────────────────────────────────────────

# Patterns that almost certainly indicate prompt injection
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I),
    re.compile(r"forget\s+(all\s+)?previous", re.I),
    re.compile(r"disregard\s+(all\s+)?", re.I),
    re.compile(r"you\s+are\s+now\s+a", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a\b)", re.I),
    re.compile(r"\bsystem\s*:\s*", re.I),
    re.compile(r"<\s*script[\s>]", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"dan\s+mode", re.I),
]


def sanitize_input(query: str) -> Tuple[bool, str, str]:
    """
    Basic regex / heuristic sanitisation.

    Returns:
        (is_valid, cleaned_query, error_message)
    """
    if not query or not query.strip():
        return False, "", "Query cannot be empty."

    query = query.strip()

    if len(query) > MAX_QUERY_LENGTH:
        return (
            False,
            "",
            f"Query too long — max {MAX_QUERY_LENGTH} characters allowed.",
        )

    # Strip control characters (keep newlines and tabs)
    query = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)

    # Injection pattern check
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            logger.warning(f"🚫 Injection pattern matched: {pattern.pattern}")
            return False, "", "Query contains disallowed content."

    return True, query, ""


# ── Tier 2 — Rate Limiting ────────────────────────────────────────────────────

def get_client_id(session_id: str) -> str:
    """Stable, hashed identifier for a session (not stored in plain text)."""
    return "client:" + hashlib.sha256(session_id.encode()).hexdigest()[:20]


def check_rate_limit(client_id: str) -> Tuple[bool, int]:
    """
    Sliding-window rate limiter using a Redis sorted set.

    Each member is a unique timestamp string; score is epoch seconds.
    Old entries (outside the window) are purged before counting.

    Returns:
        (is_allowed, remaining_requests_in_window)
    """
    if not REDIS_OK:
        # Fail open when Redis is down — log and allow
        return True, RATE_LIMIT_REQUESTS

    key = f"rl:{client_id}"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    try:
        pipe = _redis.pipeline()
        pipe.zremrangebyscore(key, "-inf", window_start)   # prune stale
        pipe.zcard(key)                                     # count in window
        pipe.zadd(key, {f"{now:.6f}": now})                # record this call
        pipe.expire(key, RATE_LIMIT_WINDOW + 1)            # auto-cleanup
        results = pipe.execute()

        count_before = results[1]   # count before we added current call

        if count_before >= RATE_LIMIT_REQUESTS:
            # Roll back the zadd we just did — this request is rejected
            _redis.zrem(key, f"{now:.6f}")
            logger.warning(f"🚦 Rate limit hit for {client_id}")
            return False, 0

        remaining = RATE_LIMIT_REQUESTS - count_before - 1
        return True, max(remaining, 0)

    except Exception as exc:
        logger.error(f"Rate limiter error: {exc}")
        return True, RATE_LIMIT_REQUESTS   # fail open


# ── Tier 3 — LLM Guard Scanning ──────────────────────────────────────────────

def scan_input(query: str) -> Tuple[bool, str, str]:
    """
    ML-based input scanning via LLM Guard (Tier 3).

    Returns:
        (is_safe, sanitized_query, reason_if_blocked)
    """
    if not ENABLE_LLM_GUARD:
        return True, query, ""

    scanners = _load_input_scanners()
    if not scanners:
        return True, query, ""

    try:
        from llm_guard import scan_prompt

        sanitized, results, is_valid = scan_prompt(scanners, query)
        if not is_valid:
            failed = [k for k, v in results.items() if not v]
            reason = f"Blocked by: {', '.join(failed)}"
            logger.warning(f"🛡️  Input scan blocked — {reason}")
            return False, sanitized, reason

        return True, sanitized, ""
    except Exception as exc:
        logger.error(f"LLM Guard input scan error: {exc}")
        return True, query, ""   # fail open


def scan_output(prompt: str, output: str) -> Tuple[bool, str, str]:
    """
    ML-based output scanning via LLM Guard (Tier 3).

    Returns:
        (is_safe, sanitized_output, reason_if_blocked)
    """
    if not ENABLE_LLM_GUARD:
        return True, output, ""

    scanners = _load_output_scanners()
    if not scanners:
        return True, output, ""

    try:
        from llm_guard import scan_output as _scan_output

        sanitized, results, is_valid = _scan_output(scanners, prompt, output)
        if not is_valid:
            failed = [k for k, v in results.items() if not v]
            reason = f"Output blocked by: {', '.join(failed)}"
            logger.warning(f"🛡️  Output scan blocked — {reason}")
            return False, sanitized, reason

        return True, sanitized, ""
    except Exception as exc:
        logger.error(f"LLM Guard output scan error: {exc}")
        return True, output, ""   # fail open


# ── Convenience: full input pipeline ─────────────────────────────────────────

def run_input_security(
    query: str, session_id: str
) -> Tuple[bool, str, str, int]:
    """
    Run all three input-security tiers in sequence.

    Returns:
        (is_allowed, cleaned_query, error_message, remaining_requests)
    """
    # Tier 1 — sanitise
    valid, query, err = sanitize_input(query)
    if not valid:
        return False, "", err, 0

    # Tier 2 — rate limit
    client_id = get_client_id(session_id)
    allowed, remaining = check_rate_limit(client_id)
    if not allowed:
        return False, "", "⚠️ Rate limit exceeded. Please wait before retrying.", 0

    # Tier 3 — LLM Guard
    safe, query, reason = scan_input(query)
    if not safe:
        return False, "", f"🛡️ Security scan failed: {reason}", remaining

    return True, query, "", remaining