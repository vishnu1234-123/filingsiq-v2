"""
scripts/inspect_concept_pick.py

Shows exactly what candidate_concepts() returned (in rank order, post-
reranking) and what _pick_concept() actually chose, for a specific
question. Built to test one specific hypothesis: does the WRONG
candidate rank ABOVE the correct one because its name is a closer
literal/lexical match to the question's own wording (a company-specific
tag written in plain English vs. a standard GAAP tag written in formal
taxonomy language)? If rank order explains the wrong pick, that's a
fixable ranking bias. If the correct concept ranks fine but gets picked
wrong anyway, that's a prompt/reasoning problem at the LLM step instead
-- two different fixes, so worth knowing which one this actually is
before touching anything.

Usage:
    python -m scripts.inspect_concept_pick xom "What was ExxonMobil's weighted-average grant-date fair value per share for outstanding restricted stock and units as of December 31, 2025?"
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("company")
    ap.add_argument("question")
    ap.add_argument("--strict-retry", action="store_true", help="use the strict_retry prompt instead of the normal one")
    args = ap.parse_args()

    from orchestration.concept_resolver import candidate_concepts
    from orchestration.router import _pick_concept

    candidates = candidate_concepts(args.company, args.question)
    print(f"Question: {args.question!r}")
    print(f"Candidates returned ({len(candidates)} total, in rank order):\n")
    for i, c in enumerate(candidates):
        print(f"  [{i}] {c}")

    print()
    picked = _pick_concept(args.question, candidates, strict_retry=args.strict_retry)
    print(f"Picked: {picked!r}")
    if picked in candidates:
        print(f"Picked candidate's rank position: {candidates.index(picked)}")


if __name__ == "__main__":
    main()