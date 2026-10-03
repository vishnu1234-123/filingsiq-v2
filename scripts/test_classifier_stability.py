"""
scripts/test_classifier_stability.py

Measures the route each borderline question ACTUALLY ends up on after
going through dispatch_subquestions -- not just classify_subquestion in
isolation. That distinction matters now: the override added to
dispatch_subquestions can send a question to "sql" even when
classify_subquestion itself still says "vector" underneath. Testing the
raw classifier alone would no longer tell you what real users actually
experience; testing dispatch_subquestions does.

Costs more per call than before -- this now does a real candidate_concepts()
lookup (no LLM cost) and, when routed to sql, a real concept-pick call
(one more LLM call) or, when routed to vector, a real retrieval call. It
does NOT run synthesis, so cost stays bounded to routing + retrieval, not
a full end-to-end answer per sample.

Usage:
    python -m scripts.test_classifier_stability
    python -m scripts.test_classifier_stability --n 20
"""

import argparse
from collections import Counter


# (question, company, expected_route) -- dispatch_subquestions needs
# company for the candidate_concepts() override check.
# First batch: confirmed vector-misclassified-as-sql cases (colloquial
# numeric phrasing that IS real SQL). Second batch: confirmed
# sql-misclassified-as-vector-topic cases -- the NEW few-shot examples
# just added to classify_subquestion's prompt, testing whether few-shot
# succeeds where the earlier abstract-rule-only prompt failed 10/10.
BORDERLINE_QUESTIONS = [
    ("Apple R&D", "aapl", "sql"),
    ("How much profit did Apple generate in 2025?", "aapl", "sql"),
    ("What was Apple's R&D spend in FY2025?", "aapl", "sql"),
    ("How much did Apple spend on research and development in FY2025?", "aapl", "sql"),
    ("Which countries individually accounted for 10% or more of Apple's net sales in FY2025?", "aapl", "vector"),
    ("What are Pfizer's rights regarding the commercialization of Braftovi and Mektovi in different regions?", "pfe", "vector"),
    ("How does ExxonMobil justify its significant investment in traditional oil and gas operations despite these risks?", "xom", "vector"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="samples per question")
    args = ap.parse_args()

    from orchestration.router import dispatch_subquestions

    print(f"Sampling actual post-override route via dispatch_subquestions, "
          f"{args.n}x per question ({len(BORDERLINE_QUESTIONS)} questions, "
          f"{args.n * len(BORDERLINE_QUESTIONS)} total calls -- costs real API money, "
          f"more per call than the old classifier-only version)\n")

    for q, company, expected in BORDERLINE_QUESTIONS:
        first_route_results = Counter()
        # dispatch_subquestions' sql-supplement fix APPENDS a vector
        # result rather than replacing the sql one -- checking only
        # sub_results[0] would miss a supplement that fired correctly.
        # Tracking both: the raw classifier decision (first route) and
        # whether real narrative content ended up ANYWHERE in the results
        # (what synthesis actually sees).
        any_vector_present = 0
        for i in range(args.n):
            state = {"sub_questions": [q], "company": company, "trace_id": f"stability-test-{i}"}
            out = dispatch_subquestions(state)
            routes = [r["route"] for r in out["sub_results"]]
            first_route_results[routes[0]] += 1
            if "vector" in routes:
                any_vector_present += 1
        print(f"{q!r} ({company}, expect={expected})")
        print(f"  raw first-route: {dict(first_route_results)}")
        print(f"  vector content present anywhere: {any_vector_present}/{args.n}"
              f"{'  (supplement is covering it)' if expected=='vector' and first_route_results.get('sql',0)>0 and any_vector_present==args.n else ''}")
        print()


if __name__ == "__main__":
    main()