from orchestration.router import answer_question,append_turn
history=None

def turn(n,question,note=""):
    global history
    print("=" * 70)
    print(f"TURN {n}{' -- ' + note if note else ''}")
    print("=" * 70)
    print(f"Asked: {question}")
    result = answer_question(question, use_cache=False, conversation_history=history)
    print(f"Company resolved: {result.get('company')}")
    print(f"Standalone question: {result.get('standalone_question')}")
    print(f"Answer: {result.get('answer')}")
    print(f"Blocked: {result.get('blocked')}, reason: {result.get('block_reason')}")
    print()
    history = append_turn(history, question, result)
    return result


# Turn 1: baseline, Apple
turn(1, "What was Apple's revenue in FY2025?")
 
# Turn 2: explicit switch to Tesla
turn(2, "What about Tesla's revenue instead?", "explicit switch, expect tsla")
 
# Turn 3: implicit follow-up right after a switch -- should use the MOST
# RECENT company (Tesla), not fall back to the older Apple mention.
turn(3, "How about their net income?", "expect tsla (most recent), NOT aapl")
 
# Turn 4: explicit switch to JPMorgan
turn(4, "What about JPMorgan's net income?", "explicit switch, expect jpm")
 
# Turn 5: implicit follow-up on the new company
turn(5, "How does that compare to their revenue?", "expect jpm")
 
# Turn 6: explicit switch BACK to Apple after several turns away --
# should work regardless of how many turns have passed, since it's explicit.
turn(6, "And Apple's margin this time?", "explicit switch back, expect aapl")
 
# Turn 7: probes the max_turns=3 window boundary directly. By now the
# window (last 3 turns) covers turns 4-6, NOT turn 1 -- so this specific
# figure should NOT be resolvable from history anymore. Watch for either
# a sensible re-query (treating it as a fresh Apple revenue question) or
# a graceful "no result" -- NOT a wrong number borrowed from the wrong turn.
turn(7, "What was that first revenue number again?", "probes 3-turn window edge -- turn 1 content may no longer be visible")