
"""
Live test -- needs a real OPENAI_API_KEY and your real DB/vectorstore set up.
Run: python -m scripts.live_test_conversational_memory
"""

from orchestration.router import answer_question,append_turn

print("=" * 70)
print("TURN 1 (no history)")
print("=" * 70)
result1=answer_question("What was Apple's revenue in FY2025?", use_cache=False)
print(f"Company resolved: {result1.get('company')}")
print(f"Standalone question: {result1.get('standalone_question')}")
print(f"Answer: {result1.get('answer')}")

history=append_turn(None,"What was Apple's revenue in FY2025?", result1)

print("\n" + "=" * 70)
print("TURN 2 (follow-up, relies on turn 1's context)")
print("=" * 70)
result2 = answer_question("How about their gross margin?", use_cache=False, conversation_history=history)
print(f"Company resolved: {result2.get('company')}")  # should be 'aapl', via history fallback or condensation
print(f"Standalone question: {result2.get('standalone_question')}")  # THIS is the line to actually judge
print(f"Answer: {result2.get('answer')}")
print(f"Blocked: {result2.get('blocked')}, reason: {result2.get('block_reason')}")
print("\n" + "=" * 70)
print("TURN 3 (explicit company switch, should override history)")
print("=" * 70)
history2 = append_turn(history, "How about their gross margin?", result2)
result3 = answer_question("What about Tesla's revenue instead?", use_cache=False, conversation_history=history2)
print(f"Company resolved: {result3.get('company')}")  # should be 'tsla', NOT 'aapl'
print(f"Standalone question: {result3.get('standalone_question')}")
print(f"Answer: {result3.get('answer')}")

