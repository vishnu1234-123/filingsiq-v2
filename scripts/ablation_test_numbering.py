"""
Isolates whether turn-numbering or just a wider window caused Turn 5's
regression. Replays the exact history from the live long-conversation test
(turns 1-4) and asks Turn 5's question through condense_question twice:
once with numbering (current code), once with numbering stripped but the
SAME window size (10) and SAME history content. Only variable that changes
is the presence of "Turn N" labels.

Run: OTEL_QUIET=true python -m scripts.ablation_test_numbering
"""
from orchestration.query_preprocessing import _model

# The real history from the live test, turns 1-4, reconstructed exactly
history = [
    {"question": "What was Apple's revenue in FY2025?",
     "answer": "Apple's revenue in FY2025 was $416,161,000,000.", "company": "aapl"},
    {"question": "What about Tesla's revenue instead?",
     "answer": "Tesla's revenue in FY2025 was $94,827,000,000.", "company": "tsla"},
    {"question": "How about their net income?",
     "answer": "Tesla's net income in FY2025 was $3,794,000,000.", "company": "tsla"},
    {"question": "What about JPMorgan's net income?",
     "answer": "JPMorgan's net income in FY2025 was $57,048,000,000.", "company": "jpm"},
]
latest_question = "How does that compare to their revenue?"

def format_numbered(history, max_turns=10):
    recent = history[-max_turns:]
    start_index = len(history) - len(recent) + 1
    lines = ["Recent conversation, numbered by turn (use the turn numbers to "
             "resolve ordinal references like 'the first question' or 'two "
             "questions ago', and use the content to resolve references "
             "like 'it', 'their', 'that', or an implied company/period/"
             "metric that isn't explicit in the CURRENT question):"]
    for offset, turn in enumerate(recent):
        n = start_index + offset
        lines.append(f"Turn {n} Q: {turn['question']}")
        lines.append(f"Turn {n} A: {turn['answer'][:300]}")
    return "\n".join(lines) + "\n\n"

def format_unnumbered(history, max_turns=10):
    # SAME window size, SAME content, SAME instruction text minus the
    # ordinal-specific sentence and minus "Turn N" labels -- isolates
    # numbering as the only variable.
    recent = history[-max_turns:]
    lines = ["Recent conversation (use this to resolve references like "
             "'it', 'their', 'that', or an implied company/period/metric "
             "that isn't explicit in the CURRENT question):"]
    for turn in recent:
        lines.append(f"Q: {turn['question']}")
        lines.append(f"A: {turn['answer'][:300]}")
    return "\n".join(lines) + "\n\n"

def build_prompt(history_block, question):
    return (
        f"{history_block}"
        f"Latest message: \"{question}\"\n\n"
        f"Rewrite the latest message as a complete, standalone question "
        f"that could be understood with NO other context -- resolve any "
        f"pronoun, implied company, implied time period, or implied topic "
        f"using the conversation above. Return ONLY the rewritten "
        f"question, nothing else."
    )

print("=" * 70)
print("VARIANT A: numbered turns (current production code)")
print("=" * 70)
prompt_a = build_prompt(format_numbered(history), latest_question)
result_a = _model.invoke(prompt_a).content.strip()
print(f"Rewrite: {result_a}")

print()
print("=" * 70)
print("VARIANT B: same window (10), same content, NO turn numbers")
print("=" * 70)
prompt_b = build_prompt(format_unnumbered(history), latest_question)
result_b = _model.invoke(prompt_b).content.strip()
print(f"Rewrite: {result_b}")

print()
print("=" * 70)
print("READ: if A says Tesla and B says JPMorgan -> numbering is the cause.")
print("      if BOTH say Tesla -> it's the wider window / more competing")
print("      mentions, independent of numbering -- removing numbers won't fix it.")
print("=" * 70)