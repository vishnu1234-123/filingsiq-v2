"""
orchestration/cache.py

Redis-backed replacement for the in-memory _EXACT_CACHE dict. Same
semantics (exact-match on company::question), but fixes what an
in-memory dict can't:

- Shared across worker processes. A plain dict is private to one process;
  the moment there's more than one uvicorn worker or replica, each has
  its own cache and hit rate craters. Redis is shared, so this just works.
- Real TTL. The dict version never expired anything -- a cached answer
  could stay "correct" long after a new 10-Q made it factually stale.
  DEFAULT_TTL_SECONDS below is a starting point, not a considered number;
  pick something that matches how often you actually re-ingest filings.
- Bounded memory, optional persistence across restarts.

Deliberately NOT semantic/similarity caching (e.g. GPTCache-style). Two
differently-worded questions ("Apple's FY2025 revenue" vs "how much did
Apple make in FY2025") are NOT treated as the same cache entry here, even
though a semantic cache might. That's intentional: for sql/compute
routes especially, a "similar" question is very often asking about a
different year or company, and a similarity threshold has no way to
tell "same question, different words" apart from "different question,
similar words" when the distinguishing token is a specific year or
ticker. Exact-match is a real cache-hit-rate cost, but a wrong-year
silent cache hit is a correctness bug with no error signal at all --
worse than the cost it would save. If semantic caching gets added later,
scope it to vector-routed narrative answers only, never sql/compute.
"""

from __future__ import annotations

import json
import os
from typing import Optional

DEFAULT_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", 60 * 60 * 24))  # 24h default -- reconsider against your real ingestion cadence
_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

_client = None


def _get_client():
    global _client
    if _client is None:
        import redis
        _client = redis.from_url(_REDIS_URL, decode_responses=True)
    return _client


def make_key(company: Optional[str], question: str) -> str:
    return f"answer_cache:{company}::{question.strip().lower()}"


def get(company: Optional[str], question: str) -> Optional[dict]:
    """Returns {"answer": ..., "faithfulness": ...} or None on a miss
    (including if Redis itself is unreachable -- a cache being down
    should degrade to "always miss," never crash the pipeline)."""
    try:
        client = _get_client()
        raw = client.get(make_key(company, question))
        return json.loads(raw) if raw else None
    except Exception:
        return None


def set(company: Optional[str], question: str, answer: str, faithfulness: Optional[float],
        ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
    """Best-effort write -- a failed cache write should never fail the
    request that produced a perfectly good answer."""
    try:
        client = _get_client()
        payload = json.dumps({"answer": answer, "faithfulness": faithfulness})
        client.setex(make_key(company, question), ttl_seconds, payload)
    except Exception:
        pass

def delete(company:Optional[str],question:str)->bool:
    try:
        client=_get_client()
        return bool(client.delete(make_key(company,question)))
    except Exception:
        return False

def invalidate_company(company: str) -> int:
    """Call this after re-ingesting a company's filing, so stale cached
    answers don't outlive the data that made them correct. Returns the
    number of keys removed."""
    try:
        client = _get_client()
        pattern = f"answer_cache:{company}::*"
        keys = list(client.scan_iter(match=pattern))
        if keys:
            client.delete(*keys)
        return len(keys)
    except Exception:
        return 0