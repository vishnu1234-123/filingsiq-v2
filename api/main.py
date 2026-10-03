"""
api/main.py

FastAPI wrapper around orchestration.router.answer_question. Conversation
history is managed SERVER-SIDE, keyed by session_id, rather than trusting
the client to send its own history back on every request -- this is what
lets a later thumbs-down feedback endpoint look up exactly which cached
answer to invalidate, and it's the more production-realistic shape.

KNOWN LIMITATION, on purpose, not hidden: _SESSIONS below is a plain
in-process dict. That's fine for local testing (single process, single
worker) but WILL break the moment this runs with more than one worker or
more than one container -- each process gets its own memory, so a session
started on one instance is invisible to another. Must be swapped for a
shared store (Redis, reusing whatever orchestration/cache.py already
connects to) before Docker/AWS run more than one instance. Flagged here
rather than silently working locally and failing mysteriously in prod.

Run: uvicorn api.main:app --reload
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from orchestration.router import answer_question, append_turn, FILENAME_MAP
from orchestration import cache as answer_cache
from orchestration.telemetry import log_decision, new_trace_id

app = FastAPI(title="FilingsIQ API", version="0.1.0")

# Resolved relative to THIS file's location, not the current working
# directory -- uvicorn can be started from different directories (project
# root locally, a container's WORKDIR later), and a bare "static" string
# has already bitten us twice tonight in other forms (scripts/ not found,
# .env not found) for exactly this reason.
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# session_id -> conversation_history (the same list[dict] shape
# append_turn/answer_question already use elsewhere in this codebase)
_SESSIONS: dict[str, list[dict]] = {}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question, as typed.")
    session_id: Optional[str] = Field(
        None, description="Omit to start a new conversation; pass back the "
        "session_id from a prior response to continue one."
    )
    use_cache: bool = True


class AskResponse(BaseModel):
    session_id: str
    company: Optional[str] = None
    standalone_question: Optional[str] = None
    answer: Optional[str] = None
    blocked: bool = False
    block_reason: Optional[str] = None
    served_from_cache: bool = False
    cost_usd: Optional[float] = None


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


class FeedbackRequest(BaseModel):
    company: str
    standalone_question: str
    thumbs_up: bool


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    # Deliberately takes company + standalone_question directly from the
    # frontend, echoed back from what /ask already returned -- rather than
    # a session_id + turn index needing a lookup into _SESSIONS. Simpler,
    # and doesn't touch the session-history code path at all.
    #
    # ASSUMPTION FLAGGED: this calls answer_cache.delete(company, question)
    # to invalidate exactly the flagged answer -- this is the earlier-
    # discussed real fix for a wrong answer otherwise being served
    # indefinitely from cache. I don't have the real cache.py's signature
    # for delete() (only a stub in my own sandbox), so if your real
    # cache.py doesn't have a delete method yet, or names its args
    # differently, this call needs adjusting to match.
    if not req.thumbs_up:
        trace_id = new_trace_id()
        answer_cache.delete(req.company, req.standalone_question)
        log_decision(
            "answer_flagged_bad", trace_id=trace_id,
            company=req.company, question=req.standalone_question,
        )
    return {"status": "recorded"}


@app.get("/companies")
def companies():
    # Single source of truth -- reads the SAME roster router.py already
    # uses for everything else, rather than a second hardcoded list here
    # or in the frontend that could silently drift out of sync with it.
    return {"companies": sorted(FILENAME_MAP.keys())}


@app.get("/health")
def health():
    # Deliberately minimal and dependency-free right now -- no DB ping, no
    # vectorstore check. Real readiness checks (can we reach Redis, can we
    # reach the vectorstore) belong here once this is containerized and an
    # AWS health check is actually load-bearing for traffic routing; a
    # premature deep health check that flaps on a transient Redis blip
    # would cause AWS to cycle healthy containers, which is worse than a
    # shallow check for now.
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    session_id = req.session_id or uuid.uuid4().hex
    if req.session_id and req.session_id not in _SESSIONS:
        # An unrecognized session_id is most likely the server having
        # restarted (in-memory store, see module docstring) rather than a
        # client bug -- fail soft by starting a fresh history under the
        # SAME id instead of a 404, so a mid-conversation server restart
        # degrades to "forgot the context" rather than a hard error.
        _SESSIONS[req.session_id] = []

    history = _SESSIONS.get(session_id)
    result = answer_question(req.question, use_cache=req.use_cache, conversation_history=history)
    _SESSIONS[session_id] = append_turn(history, req.question, result)

    cost_summary = result.get("cost_summary") or {}
    return AskResponse(
        session_id=session_id,
        company=result.get("company"),
        standalone_question=result.get("standalone_question"),
        answer=result.get("answer"),
        blocked=result.get("blocked", False),
        block_reason=result.get("block_reason"),
        served_from_cache=result.get("served_from_cache", False),
        cost_usd=cost_summary.get("total_cost_usd"),
    )


@app.get("/session/{session_id}")
def get_session(session_id: str):
    # Mainly for debugging/testing right now -- lets you see what history
    # the server actually thinks a session has, without needing to trust
    # client-side state while building the frontend next.
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id, "history": _SESSIONS[session_id]}