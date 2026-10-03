"""
scripts/debug_ms_narr_34.py

Checks whether decomposition is genuinely producing an identical
duplicate sub-question, or whether two visually-identical strings
actually differ by hidden whitespace/characters (which would explain why
the dedup fix -- confirmed present in router.py -- isn't catching this
case: it uses exact string equality).

Usage:
    python -m scripts.debug_ms_narr_34
"""

from orchestration.router import preprocess_query_node, dispatch_subquestions


def main():
    state = {
        "question": "How does Morgan Stanley's ability to access capital affect its competitive position in the market?",
        "company": "ms",
        "trace_id": "debug1",
    }
    analysis_state = preprocess_query_node(state)
    subqs = analysis_state["sub_questions"]

    print(f"DECOMPOSITION: {len(subqs)} sub-question(s):")
    for i, q in enumerate(subqs):
        print(f"  [{i}] repr={q!r}")

    # 5/5 prior runs confirmed decomposition alone always produces exactly
    # 1 sub-question here -- so if the real eval showed 2 sub_results with
    # both vector and sql routes, the duplication has to happen somewhere
    # INSIDE dispatch_subquestions itself, not at decomposition. Running
    # the real dispatch step (real classify_subquestion, real
    # candidate_concepts against the real DB) to see exactly how many
    # sub_results come out and what routes they take.
    print()
    print("DISPATCH (real classify_subquestion + candidate_concepts):")
    dispatch_state = dispatch_subquestions(analysis_state)
    sub_results = dispatch_state["sub_results"]
    print(f"{len(sub_results)} sub_result(s):")
    for i, r in enumerate(sub_results):
        print(f"  [{i}] route={r['route']!r} status={r['status']!r} question={r['sub_question']!r}")


if __name__ == "__main__":
    main()