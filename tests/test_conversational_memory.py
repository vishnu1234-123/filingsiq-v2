"""
Pytest-native version. Run: OTEL_QUIET=true pytest tests/test_conversational_memory.py -v
"""
import os
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")

from unittest.mock import patch
from orchestration.router import entity_check, append_turn, condense_query_node
from orchestration.query_preprocessing import is_trivially_simple, _format_history, condense_question


# ---- condense_question ----

def test_condense_question_noop_without_history():
    with patch("orchestration.query_preprocessing._model") as mock_model:
        result = condense_question("What was Apple's revenue in FY2025?", None)
        assert result == "What was Apple's revenue in FY2025?"
        assert mock_model.invoke.call_count == 0  # no wasted call on turn 1


def test_condense_question_invokes_llm_with_history():
    with patch("orchestration.query_preprocessing._model") as mock_model:
        mock_model.invoke.return_value.content = "What was Apple's gross margin in FY2025?"
        result = condense_question(
            "How about their gross margin?",
            [{"question": "What was Apple's revenue in FY2025?", "answer": "It was $416B.", "company": "aapl"}],
        )
        assert mock_model.invoke.call_count == 1
        assert result == "What was Apple's gross margin in FY2025?"


def test_condense_question_prompt_includes_history_context():
    with patch("orchestration.query_preprocessing._model") as mock_model:
        mock_model.invoke.return_value.content = "irrelevant"
        condense_question(
            "How about their gross margin?",
            [{"question": "What was Apple's revenue in FY2025?", "answer": "It was $416B.", "company": "aapl"}],
        )
        prompt_sent = mock_model.invoke.call_args[0][0]
        assert "What was Apple's revenue" in prompt_sent
        assert "$416B" in prompt_sent


# ---- condense_query_node ----

def test_condense_query_node_sets_standalone_question():
    with patch("orchestration.router.condense_question", return_value="What was Apple's gross margin in FY2025?"):
        state = {
            "question": "How about their gross margin?",
            "conversation_history": [{"question": "...", "answer": "...", "company": "aapl"}],
            "trace_id": "t3",
        }
        result = condense_query_node(state)
        assert result["standalone_question"] == "What was Apple's gross margin in FY2025?"
        assert result["question"] == "How about their gross margin?"  # original preserved


def test_condense_query_node_noop_without_history():
    with patch("orchestration.router.condense_question", return_value="What was JPM's net income?"):
        state = {"question": "What was JPM's net income?", "conversation_history": None, "trace_id": "t4"}
        result = condense_query_node(state)
        assert result["standalone_question"] == result["question"]


# ---- entity_check ----

def test_entity_check_no_company_no_history_blocks():
    """Regression: unchanged behavior with no history at all."""
    with patch("orchestration.router.resolve_company", return_value=None), \
         patch("orchestration.router.is_out_of_domain", return_value=False), \
         patch("orchestration.router.is_garbage_input", return_value=False):
        state = {"question": "How about their gross margin?", "trace_id": "t1", "conversation_history": None}
        result = entity_check(state)
        assert result["blocked"] is True
        assert result["block_reason"] == "ambiguous_or_missing_company"


def test_entity_check_falls_back_to_history_company():
    history = [{"question": "What was Apple's revenue in FY2025?", "answer": "...", "company": "aapl"}]
    with patch("orchestration.router.resolve_company", return_value=None), \
         patch("orchestration.router.is_out_of_domain", return_value=False), \
         patch("orchestration.router.is_garbage_input", return_value=False):
        state = {"question": "How about their gross margin?", "trace_id": "t1", "conversation_history": history}
        result = entity_check(state)
        assert result["blocked"] is False
        assert result["company"] == "aapl"


def test_entity_check_explicit_company_overrides_history():
    history = [{"question": "What was Apple's revenue in FY2025?", "answer": "...", "company": "aapl"}]
    with patch("orchestration.router.resolve_company", return_value="tsla"), \
         patch("orchestration.router.is_out_of_domain", return_value=False), \
         patch("orchestration.router.is_garbage_input", return_value=False):
        state = {"question": "What about Tesla's revenue?", "trace_id": "t1", "conversation_history": history}
        result = entity_check(state)
        assert result["blocked"] is False
        assert result["company"] == "tsla"


def test_entity_check_stateless_call_regression():
    with patch("orchestration.router.resolve_company", return_value="jpm"), \
         patch("orchestration.router.is_out_of_domain", return_value=False), \
         patch("orchestration.router.is_garbage_input", return_value=False):
        state = {"question": "What was JPM's net income?", "trace_id": "t2", "conversation_history": None}
        result = entity_check(state)
        assert result["blocked"] is False
        assert result["company"] == "jpm"


def test_entity_check_uses_standalone_question_for_resolution():
    def resolve_only_if_standalone(q):
        return "aapl" if "Apple" in q else None

    with patch("orchestration.router.resolve_company", side_effect=resolve_only_if_standalone), \
         patch("orchestration.router.is_out_of_domain", return_value=False), \
         patch("orchestration.router.is_garbage_input", return_value=False):
        state = {
            "question": "How about their gross margin?",  # raw -- would fail alone
            "standalone_question": "What was Apple's gross margin in FY2025?",  # condensed -- names Apple
            "conversation_history": None, "trace_id": "t5",
        }
        result = entity_check(state)
        assert result["blocked"] is False
        assert result["company"] == "aapl"


# ---- is_trivially_simple ----

def test_is_trivially_simple_short_question():
    assert is_trivially_simple("What was Apple's net income?") is True


def test_is_trivially_simple_long_question():
    assert is_trivially_simple(
        "How does Apple's approach to litigation reserves compare across its "
        "different reportable operating segments this fiscal year?"
    ) is False


# ---- _format_history ----

def test_format_history_empty():
    assert _format_history(None) == ""
    assert _format_history([]) == ""


def test_format_history_includes_content():
    formatted = _format_history([
        {"question": "What was Apple's revenue?", "answer": "Apple's FY2025 revenue was $416B."},
    ])
    assert "What was Apple's revenue?" in formatted
    assert "$416B" in formatted


def test_format_history_respects_max_turns():
    long_history = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(5)]
    formatted = _format_history(long_history, max_turns=3)
    assert "Q0" not in formatted
    assert "Q4" in formatted


# ---- append_turn ----

def test_append_turn_from_empty():
    h1 = append_turn(None, "What was Apple's revenue?", {"answer": "It was $416B.", "company": "aapl"})
    assert h1 == [{"question": "What was Apple's revenue?", "answer": "It was $416B.", "company": "aapl"}]


def test_append_turn_preserves_and_appends():
    h1 = append_turn(None, "What was Apple's revenue?", {"answer": "It was $416B.", "company": "aapl"})
    h2 = append_turn(h1, "How about their gross margin?", {"answer": "46.2%.", "company": "aapl"})
    assert len(h2) == 2
    assert h2[0] == h1[0]
    assert h2[1]["question"] == "How about their gross margin?"


def test_append_turn_no_mutation():
    h1 = append_turn(None, "What was Apple's revenue?", {"answer": "It was $416B.", "company": "aapl"})
    _ = append_turn(h1, "How about their gross margin?", {"answer": "46.2%.", "company": "aapl"})
    assert len(h1) == 1  # h1 itself must be untouched


def test_append_turn_records_blocked_turns():
    h1 = append_turn(None, "What was Apple's revenue?", {"answer": "It was $416B.", "company": "aapl"})
    blocked_result = {"answer": None, "company": "aapl", "blocked": True}
    h3 = append_turn(h1, "asdkjasdkj garbage", blocked_result)
    assert len(h3) == 2
    assert h3[1]["answer"] is None
    assert h3[1]["company"] == "aapl"