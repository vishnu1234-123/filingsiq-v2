"""
Run: export PHOENIX_COLLECTOR_ENDPOINT="http://localhost:6006"
     python -m scripts.phoenix_sample_traces
Then check http://localhost:6006 for a handful of differently-shaped traces.
"""
from orchestration.router import answer_question, append_turn

print("1. SQL-style factual lookup...")
r1 = answer_question("What was Apple's net income in FY2025?", use_cache=False)
print(f"   -> {r1.get('answer')}\n")

print("2. Narrative/vector-style reasoning question...")
r2 = answer_question("Why does Tesla describe its reliance on Elon Musk as a risk factor?", use_cache=False)
print(f"   -> {r2.get('answer')}\n")

print("3. Conversational follow-up (exercises condense_question too)...")
history = append_turn(None, "What was Apple's net income in FY2025?", r1)
r3 = answer_question("How does that compare to their revenue?", use_cache=False, conversation_history=history)
print(f"   -> {r3.get('answer')}\n")

print("Done -- check http://localhost:6006 for 3 traces.")
print("Look for: prompt/completion text rendered on the LLM spans (not blank),")
print("token counts, and the node graph shape differing between the SQL")
print("question (1) and the narrative question (2).")