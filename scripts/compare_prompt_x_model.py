"""
scripts/compare_prompt_x_model.py

Earlier tonight, adding 4 new few-shot examples to classify_subquestion
regressed SQL accuracy on gpt-4o-mini (3/5 -> 0/5 on a small batch) and
was reverted. Open question: was that a bad prompt, or a model-capability
limit that a stronger model doesn't share? This runs baseline vs
expanded prompt, crossed with two models, on the real golden set, and
reports whether the expansion helps, hurts, or does nothing under each.

Does not modify router.py. Rebuilds the prompt string locally and calls
MODEL directly, bypassing classify_subquestion, so nothing here can
accidentally leave a change in the live classifier.

Usage:
    python -m scripts.compare_prompt_x_model --models gpt-4o-mini gpt-4o --n 3
"""

import argparse
import json
import re
from collections import Counter

from langchain_openai import ChatOpenAI

BASE_PROMPT_TAIL = (
    "Confirmed examples of questions that LOOK numeric but are VECTOR, not "
    "SQL, because the answer is only ever disclosed in narrative or table "
    "text, not as one discrete tagged fact:\n"
    "- \"Which countries individually accounted for 10% or more of Apple's "
    "net sales?\" -> VECTOR (a per-country breakdown is disclosed in an "
    "MD&A table, not one XBRL fact per country)\n"
    "- \"What are Pfizer's rights regarding Braftovi and Mektovi in "
    "different regions?\" -> VECTOR (a licensing/rights structure across "
    "regions is narrative, never a filing tag)\n"
    "- \"How does ExxonMobil justify its investment in oil and gas despite "
    "climate risk?\" -> VECTOR (\"justify\" asks for the company's own "
    "reasoning/framing, not a number)\n"
    "- \"What does Pfizer's filing say about integration risks?\" -> VECTOR "
    "(\"what does the filing say about\" asks for disclosed prose)\n"
    "- \"What was Apple's R&D spend in FY2025?\" -> SQL (a real GAAP line "
    "item, just phrased colloquially -- 'spend' means 'expense' here)\n\n"
    "Reply with one word."
)

EXPANDED_PROMPT_TAIL = (
    "Confirmed examples of questions that LOOK numeric but are VECTOR, not "
    "SQL, because the answer is only ever disclosed in narrative or table "
    "text, not as one discrete tagged fact:\n"
    "- \"Which countries individually accounted for 10% or more of Apple's "
    "net sales?\" -> VECTOR (a per-country breakdown is disclosed in an "
    "MD&A table, not one XBRL fact per country)\n"
    "- \"What are Pfizer's rights regarding Braftovi and Mektovi in "
    "different regions?\" -> VECTOR (a licensing/rights structure across "
    "regions is narrative, never a filing tag)\n"
    "- \"How does ExxonMobil justify its investment in oil and gas despite "
    "climate risk?\" -> VECTOR (\"justify\" asks for the company's own "
    "reasoning/framing, not a number)\n"
    "- \"What does Pfizer's filing say about integration risks?\" -> VECTOR "
    "(\"what does the filing say about\" asks for disclosed prose)\n\n"
    "Confirmed examples of questions that ARE SQL even though they are "
    "phrased casually or as fragments, because each one names a single "
    "standard financial-statement line item:\n"
    "- \"Tesla R&D\" -> SQL (a fragment naming one line item: research and "
    "development expense)\n"
    "- \"What was Caterpillar's R&D spend in 2024?\" -> SQL ('spend' means "
    "'expense' here)\n"
    "- \"How much profit did Tesla make in 2025?\" -> SQL ('profit' means "
    "net income)\n"
    "- \"How much did Walmart spend on advertising in fiscal 2025?\" -> SQL "
    "(advertising expense)\n\n"
    "Reply with one word."
)

PROMPT_HEAD = (
    "Question: {question}\n\n"
    "Classify as exactly one of:\n"
    "SQL - a specific figure that would appear as a GAAP/XBRL financial "
    "statement line item (revenue, assets, liabilities, shares outstanding, "
    "per-share metrics, tax rates, etc.)\n"
    "COMPUTE - a trend or growth rate across multiple periods of such a figure\n"
    "VECTOR - narrative content, including qualitative claims that happen to "
    "sound numeric but would never be a structured filing tag (market share, "
    "competitor comparisons, industry statistics, analyst estimates, or "
    "anything a filing would only ever discuss in prose, not disclose as data)\n\n"
)

PROMPTS = {"baseline": BASE_PROMPT_TAIL, "expanded": EXPANDED_PROMPT_TAIL}


def classify(model, question, tail):
    prompt = PROMPT_HEAD.format(question=question) + tail
    decision = model.invoke(prompt).content.strip().upper()
    if "COMPUTE" in decision:
        return "compute"
    if "SQL" in decision:
        return "sql"
    return "vector"


def run_cell(model_name, prompt_label, n, golden_file):
    model = ChatOpenAI(model=model_name, temperature=0)
    tail = PROMPTS[prompt_label]
    with open(golden_file) as f:
        rows = json.load(f)
    expected_by_type = {"sql_only": "sql", "vector_only": "vector"}
    by_type = Counter()
    correct = Counter()
    for r in rows:
        expected = expected_by_type.get(r.get("type"))
        if not expected or "," in r.get("company", ""):
            continue
        votes = Counter(classify(model, r["question"], tail) for _ in range(n))
        by_type[expected] += 1
        if votes.get(expected, 0) / n >= 0.8:
            correct[expected] += 1
        print(".", end="", flush=True)
    print()
    return correct["sql"], by_type["sql"], correct["vector"], by_type["vector"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--golden-file", default="eval/golden_set_core.json")
    args = ap.parse_args()

    results = {}
    for model_name in args.models:
        for prompt_label in ("baseline", "expanded"):
            print(f"--- {model_name} / {prompt_label}")
            results[(model_name, prompt_label)] = run_cell(model_name, prompt_label, args.n, args.golden_file)

    print()
    print(f"{'model':16} {'prompt':10} {'sql':>10} {'vector':>10}")
    for (model_name, prompt_label), (sc, st, vc, vt) in results.items():
        print(f"{model_name:16} {prompt_label:10} {sc:>4}/{st:<4} {vc:>4}/{vt:<4}")

    print()
    for model_name in args.models:
        base_sc, base_st, _, _ = results[(model_name, "baseline")]
        exp_sc, exp_st, _, _ = results[(model_name, "expanded")]
        delta = exp_sc - base_sc
        verdict = "HELPS" if delta > 0 else ("HURTS" if delta < 0 else "NO CHANGE")
        print(f"{model_name}: baseline {base_sc}/{base_st} -> expanded {exp_sc}/{exp_st}  ({verdict}, delta={delta:+d})")


if __name__ == "__main__":
    main()