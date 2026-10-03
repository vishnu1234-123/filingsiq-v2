import os
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")

from unittest.mock import patch
from fastapi.testclient import TestClient
from api.main import app, _SESSIONS

client = TestClient(app)


def _fake_result(company, standalone_question, answer, **extra):
    base = {
        "company": company, "standalone_question": standalone_question,
        "answer": answer, "blocked": False, "block_reason": None,
        "served_from_cache": False, "cost_summary": {"total_cost_usd": 0.001},
    }
    base.update(extra)
    return base


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_without_session_id_creates_new_session():
    _SESSIONS.clear()
    with patch("api.main.answer_question", return_value=_fake_result(
        "aapl", "What was Apple's revenue in FY2025?", "Apple's revenue was $416,161,000,000."
    )):
        r = client.post("/ask", json={"question": "What was Apple's revenue in FY2025?"})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"]  # a new one was generated
    assert body["company"] == "aapl"
    assert body["answer"] == "Apple's revenue was $416,161,000,000."
    assert body["session_id"] in _SESSIONS  # history was actually stored


def test_ask_with_session_id_passes_history_to_answer_question():
    _SESSIONS.clear()
    sid = "test-session-1"
    _SESSIONS[sid] = [{"question": "What was Apple's revenue in FY2025?",
                        "answer": "It was $416,161,000,000.", "company": "aapl"}]

    with patch("api.main.answer_question", return_value=_fake_result(
        "aapl", "What was Apple's gross margin in FY2025?", "46.9%."
    )) as mock_answer:
        r = client.post("/ask", json={"question": "How about their gross margin?", "session_id": sid})

    assert r.status_code == 200
    # confirm the EXISTING history was actually passed through, not dropped
    _, kwargs = mock_answer.call_args
    assert kwargs["conversation_history"] == _SESSIONS[sid] or mock_answer.call_args[0]
    call_args, call_kwargs = mock_answer.call_args
    passed_history = call_kwargs.get("conversation_history", call_args[2] if len(call_args) > 2 else None)
    assert passed_history is not None and len(passed_history) == 1
    assert passed_history[0]["company"] == "aapl"


def test_ask_appends_to_existing_session_history():
    _SESSIONS.clear()
    sid = "test-session-2"
    _SESSIONS[sid] = [{"question": "Q1", "answer": "A1", "company": "aapl"}]

    with patch("api.main.answer_question", return_value=_fake_result("aapl", "Q2 standalone", "A2")):
        r = client.post("/ask", json={"question": "Q2", "session_id": sid})

    assert r.status_code == 200
    assert len(_SESSIONS[sid]) == 2  # appended, not replaced
    assert _SESSIONS[sid][0]["question"] == "Q1"  # original turn preserved
    assert _SESSIONS[sid][1]["question"] == "Q2"


def test_ask_with_unrecognized_session_id_fails_soft():
    """An unknown session_id (e.g. after a server restart) should start a
    fresh history under the SAME id, not 404 -- see module docstring on
    the in-memory session store limitation."""
    _SESSIONS.clear()
    unknown_sid = "never-seen-before"
    with patch("api.main.answer_question", return_value=_fake_result("aapl", "Q", "A")):
        r = client.post("/ask", json={"question": "Q", "session_id": unknown_sid})
    assert r.status_code == 200
    assert r.json()["session_id"] == unknown_sid  # kept the SAME id, didn't silently swap it
    assert unknown_sid in _SESSIONS


def test_get_session_returns_history():
    _SESSIONS.clear()
    sid = "test-session-3"
    _SESSIONS[sid] = [{"question": "Q1", "answer": "A1", "company": "aapl"}]
    r = client.get(f"/session/{sid}")
    assert r.status_code == 200
    assert r.json()["history"] == _SESSIONS[sid]


def test_get_session_404_for_unknown_id():
    _SESSIONS.clear()
    r = client.get("/session/does-not-exist")
    assert r.status_code == 404


def test_ask_rejects_empty_question():
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422  # pydantic min_length validation

def test_index_serves_html():
    r=client.get("/")
    assert r.status_code==200
    assert "FilingsIQ" in r.text

def test_feedback_thumbs_up_does_not_touch_cache():
    with patch("api.main.answer_cache.delete") as mock_delete:
        r=client.post("/feedback",json={
            "company":"aapl","standalone_question":"What was Apple's revenue in FY2025?",
            "thumbs_up":True,
        })
    assert r.status_code==200
    assert r.json()=={"status":"recorded"}
    mock_delete.assert_not_called()

def test_feedback_thumbs_down_invalidates_cache():
    with patch("api.main.answer_cache.delete") as mock_delete:
        r=client.post("/feedback",json={
            "company":"aapl","standalone_question":"What was Apple's revenue in FY2025?",
            "thumbs_up":False,
        })
    assert r.status_code==200
    mock_delete.assert_called_once_with("aapl","What was Apple's revenue in FY2025?")

def test_ask_surfaces_blocked_result():
    _SESSIONS.clear()
    with patch("api.main.answer_question", return_value=_fake_result(
        None, None, None, blocked=True, block_reason="ambiguous_or_missing_company"
    )):
        r = client.post("/ask", json={"question": "asdkjasdkj"})
    assert r.status_code == 200  # blocked is a normal response, not an HTTP error
    body = r.json()
    assert body["blocked"] is True
    assert body["block_reason"] == "ambiguous_or_missing_company"