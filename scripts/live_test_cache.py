"""
Confirms the standalone_question cache-key fix actually works end to end:
run the same two-turn conversation twice. Pass 1 should be two full misses.
Pass 2, with fresh conversation_history (new session) but the SAME
questions asked, should hit cache on both turns -- proving cache_check and
save_to_cache are keying consistently off standalone_question.

Run: OTEL_QUIET=true python -m scripts.live_test_cache
"""
import time
from orchestration.router import answer_question, append_turn

def run_pass(pass_label):
    print("=" * 70)
    print(pass_label)
    print("=" * 70)

    t0 = time.time()
    r1 = answer_question("What was Apple's revenue in FY2025?", use_cache=True)
    t1 = time.time()
    print(f"Turn 1 -- served_from_cache: {r1.get('served_from_cache')}, "
          f"took {t1 - t0:.2f}s")
    print(f"  Company: {r1.get('company')}, Standalone: {r1.get('standalone_question')}")
    print(f"  Answer: {r1.get('answer')}")

    history = append_turn(None, "What was Apple's revenue in FY2025?", r1)

    t2 = time.time()
    r2 = answer_question("How about their gross margin?", use_cache=True, conversation_history=history)
    t3 = time.time()
    print(f"Turn 2 -- served_from_cache: {r2.get('served_from_cache')}, "
          f"took {t3 - t2:.2f}s")
    print(f"  Company: {r2.get('company')}, Standalone: {r2.get('standalone_question')}")
    print(f"  Answer: {r2.get('answer')}")
    print()
    return r1, r2

print("PASS 1 -- expect BOTH turns to be full misses (served_from_cache: False)\n")
run_pass("PASS 1 (fresh, should be cache MISSES)")

print("PASS 2 -- same two questions, fresh conversation_history (new session).")
print("Expect BOTH turns to be cache HITS (served_from_cache: True), and")
print("noticeably faster than Pass 1 (though not instant -- condensation")
print("still runs on Turn 2 before the cache lookup happens).\n")
run_pass("PASS 2 (replay, should be cache HITS)")