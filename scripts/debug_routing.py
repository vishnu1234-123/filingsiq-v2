"""
scripts/debug_routing.py

Shows, for any question: what decomposition produced, then what
dispatch did with each sub-question -- including WHERE each result
came from (origin), so genuine decomposition can be told apart from
supplemental evidence added for the same sub-question.

Usage:
    python -m scripts.debug_routing ms "How does Morgan Stanley's ability to access capital affect its competitive position?"
    python -m scripts.debug_routing aapl "What was Apple's total revenue in fiscal 2025, and what risks does Apple describe about its supply chain?"
"""

import argparse

from orchestration.router import preprocess_query_node, dispatch_subquestions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("company")
    ap.add_argument("question")
    args = ap.parse_args()

    state = {"question": args.question, "company": args.company, "trace_id": "debug-routing"}
    analysis_state = preprocess_query_node(state)
    subqs = analysis_state["sub_questions"]

    print(f"DECOMPOSITION: {len(subqs)} sub-question(s)")
    for i, q in enumerate(subqs):
        print(f"  [{i}] {q!r}")

    print()
    dispatch_state = dispatch_subquestions(analysis_state)
    results = dispatch_state["sub_results"]
    print(f"DISPATCH: {len(results)} result(s)")
    for i, r in enumerate(results):
        print(f"  [{i}] route={r['route']!r:9} origin={r.get('origin')!r}")
        print(f"       question={r['sub_question']!r}")


if __name__ == "__main__":
    main()