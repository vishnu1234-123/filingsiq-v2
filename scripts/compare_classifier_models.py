"""
scripts/compare_classifier_models.py

Tests whether swapping the model classify_subquestion runs on closes the
gap we measured tonight: 23/23 correct on vector rows, 7/24 correct on
sql rows, on the real golden set. Before considering fine-tuning, this
answers the cheap question first -- does a stronger off-the-shelf model
already fix it?

This does NOT touch orchestration/router.py or any live pipeline
behavior. It monkey-patches router.MODEL for the duration of the test
only, in this process, then restores it.

NOTE: adjust the model construction below if your MODEL is not built
with langchain_openai.ChatOpenAI(model=..., temperature=0) -- match
however router.py actually constructs it.

Usage:
    python -m scripts.compare_classifier_models --models gpt-4o-mini gpt-4o --n 3
    python -m scripts.compare_classifier_models --models gpt-4o-mini gpt-4o --n 3 --golden-file eval/golden_set_core.json
"""

import argparse
import json
from collections import Counter

from langchain_openai import ChatOpenAI

import orchestration.router as router


def run_one_model(model_name: str, n: int, golden_file: str):
    original_model = router.MODEL
    router.MODEL = ChatOpenAI(model=model_name, temperature=0)
    try:
        with open(golden_file) as f:
            rows = json.load(f)
        expected_by_type = {"sql_only": "sql", "vector_only": "vector"}

        by_type = Counter()
        correct_by_type = Counter()
        wrong_examples = {"sql": [], "vector": []}
        wrong_ids = {"sql": set(), "vector": set()}

        for r in rows:
            expected = expected_by_type.get(r.get("type"))
            if not expected or "," in r.get("company", ""):
                continue
            votes = Counter(router.classify_subquestion(r["question"]) for _ in range(n))
            rate = votes.get(expected, 0) / n
            by_type[expected] += 1
            if rate >= 0.8:
                correct_by_type[expected] += 1
            else:
                wrong_ids[expected].add(r["id"])
                if len(wrong_examples[expected]) < 10:
                    wrong_examples[expected].append((r["id"], dict(votes)))
            print(".", end="", flush=True)

        print()
        print(f"MODEL: {model_name}")
        for t in ("sql", "vector"):
            print(f"    {t:6}: {correct_by_type[t]}/{by_type[t]} correct")
            for row_id, votes in wrong_examples[t]:
                print(f"        wrong: {row_id:18} votes={votes}")
        print()
        return {"model": model_name, "sql": (correct_by_type["sql"], by_type["sql"]),
                "vector": (correct_by_type["vector"], by_type["vector"]),
                "wrong_sql_ids": wrong_ids["sql"]}
    finally:
        router.MODEL = original_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="model names to compare, e.g. gpt-4o-mini gpt-4o")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--golden-file", default="eval/golden_set_core.json")
    args = ap.parse_args()

    results = [run_one_model(m, args.n, args.golden_file) for m in args.models]

    print("COMPARISON")
    print(f"{'model':20} {'sql':>10} {'vector':>10}")
    for r in results:
        sql_c, sql_t = r["sql"]
        vec_c, vec_t = r["vector"]
        print(f"{r['model']:20} {sql_c:>4}/{sql_t:<4} {vec_c:>4}/{vec_t:<4}")

    # Rows the gated override was measured to rescue, on the golden set,
    # regardless of which underlying model classify_subquestion uses --
    # checking whether the strongest model's remaining failures overlap
    # with this set (already covered) or are a NEW, different problem.
    # Full rescued set from the real golden-set run (gaap_vocab OR
    # figure-request wording), 11 rows -- NOT the smaller 6-row set from
    # an earlier, narrower offline test. Using the smaller set here
    # first gave a wrong "new, uncovered" verdict on jpm_sql_3b,
    # ms_sql_17 and pfe_sql_8, which the real gate already covers.
    GATE_RESCUED = {
        "jpm_sql_3", "pfe_sql_7", "pfe_sql_8", "wmt_sql_9", "cat_sql_13",
        "cat_sql_14", "met_sql_15", "ms_sql_17", "tsla_sql_20", "wmt_sql_9b",
        "jpm_sql_3b",
    }
    # Confirmed NOT rescued by the gate in that same run -- the real,
    # remaining hard cases regardless of which model classifies them.
    GATE_NOT_RESCUED = {"jpm_sql_4", "xom_sql_6", "pld_sql_12", "met_sql_16", "ms_sql_18", "tsla_sql_19"}
    if len(results) >= 2:
        strongest = max(results, key=lambda r: r["sql"][0])
        still_wrong = strongest["wrong_sql_ids"]
        print()
        print(f"OVERLAP CHECK ({strongest['model']}'s remaining {len(still_wrong)} wrong sql rows vs. gate-rescued set)")
        print(f"    already covered by the gate: {sorted(still_wrong & GATE_RESCUED)}")
        print(f"    NEW rows the gate does NOT cover: {sorted(still_wrong - GATE_RESCUED)}")


if __name__ == "__main__":
    main()