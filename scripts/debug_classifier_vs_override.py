"""
scripts/debug_classifier_vs_override.py

Measures the prompt-based classify_subquestion and simulates the
candidate_concepts override -- current (fires on ANY candidate) vs
proposed (gated by looks_like_figure_request) -- against the real
golden set. Supports --model to swap which model classify_subquestion
runs on for this test only (router.MODEL is restored after), so
"model X combined with the gate" can be measured directly instead of
composed by hand from two separate, incomplete tests.

Whole golden set, current router.MODEL, gate simulation:
    python -m scripts.debug_classifier_vs_override --golden --n 3

Whole golden set, a SPECIFIC model, gate simulation (the actual
"model + gate combined" number):
    python -m scripts.debug_classifier_vs_override --golden --n 3 --model gpt-4o
"""

import argparse
import json
from collections import Counter

from langchain_openai import ChatOpenAI

import orchestration.router as router
from orchestration.concept_resolver import candidate_concepts, looks_like_figure_request
from orchestration.router import classify_subquestion

OK_RATE = 0.8
CORRECT = "classifier correct"
FLIPPED = "OVERRIDE FLIPS a correct vector verdict to sql"
RESCUED = "classifier wrong, override rescues it"
NOT_RESCUED = "classifier wrong, NOT rescued"
WRONG = "classifier wrong"


def categorize(expected, rate, override_fires):
    if rate >= OK_RATE:
        return FLIPPED if (expected == "vector" and override_fires) else CORRECT
    if expected == "sql":
        return RESCUED if override_fires else NOT_RESCUED
    return WRONG


def assess(company, question, expected, n, classify_fn=None):
    classify_fn = classify_fn or classify_subquestion
    votes = Counter(classify_fn(question) for _ in range(n))
    candidates = candidate_concepts(company, question)
    gate = looks_like_figure_request(question)
    rate = votes.get(expected, 0) / n
    return {
        "votes": dict(votes),
        "n_cands": len(candidates),
        "gate": gate,
        "current": categorize(expected, rate, bool(candidates)),
        "proposed": categorize(expected, rate, bool(candidates) and gate),
    }


def print_summary(records):
    for key, title in (("current", "CURRENT override"), ("proposed", "PROPOSED override (gated)")):
        tallies = Counter((exp, a[key]) for _, exp, a in records)
        print(title)
        for (exp, cat), count in sorted(tallies.items()):
            print(f"    expect={exp:6} {cat}: {count}")
        print()


def run_golden(n, path, model_name=None):
    """
    dispatch classifies the decomposed/rewritten sub-question, this uses
    the original question text, so treat results as a close
    approximation, not an exact replay of live routing.
    """
    classify_fn = None
    original_model = router.MODEL
    if model_name:
        router.MODEL = ChatOpenAI(model=model_name, temperature=0)
        classify_fn = router.classify_subquestion  # picks up the swapped MODEL

    try:
        with open(path) as f:
            rows = json.load(f)
        expected_by_type = {"sql_only": "sql", "vector_only": "vector"}
        records = []
        for r in rows:
            expected = expected_by_type.get(r.get("type"))
            if not expected or "," in r.get("company", ""):
                continue
            records.append((r["id"], expected, assess(r["company"], r["question"], expected, n, classify_fn)))
            print(".", end="", flush=True)
        print("\n")
        if model_name:
            print(f"MODEL FOR THIS RUN: {model_name}\n")
        print_summary(records)
        print("ROWS NEEDING ATTENTION UNDER THE PROPOSED OVERRIDE:")
        for rid, exp, a in records:
            if a["proposed"] != CORRECT:
                print(f"    {rid:22} expect={exp:6} votes={a['votes']} candidates={a['n_cands']} gate={a['gate']}")
                print(f"        -> {a['proposed']}")
    finally:
        router.MODEL = original_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", action="store_true")
    ap.add_argument("--golden-file", default="eval/golden_set_core.json")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--model", default=None, help="swap classify_subquestion's model for this run only")
    args = ap.parse_args()
    if not args.golden:
        ap.error("this rebuilt version only supports --golden; see conversation history for --batch")
    run_golden(args.n, args.golden_file, args.model)


if __name__ == "__main__":
    main()