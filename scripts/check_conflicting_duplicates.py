"""
scripts/check_conflicting_duplicates.py

Checks golden_set_draft.json for ids that appear more than once with
DIFFERENT question text -- as opposed to exact repeats. This matters
specifically because recent manual edits (aapl_contradiction_56,
xom_contradiction_58, pfe_contradiction_59) may coexist with older,
pre-edit copies of the same id from the earlier duplication bug. A
naive dedup (keep first occurrence) could silently keep the WRONG
(old, unedited) version depending on where each copy landed in the
file. Run this BEFORE running scripts.dedupe_golden_set.

Usage:
    python -m scripts.check_conflicting_duplicates
"""

import json
from collections import defaultdict


def main():
    rows = json.loads(open("eval/golden_set_draft.json").read())
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["id"]].append(r)

    differing = {
        id_: entries for id_, entries in by_id.items()
        if len(entries) > 1 and len(set(e["question"] for e in entries)) > 1
    }

    print(f"{len(rows)} total rows, {len(by_id)} distinct ids")
    print(f"{len(differing)} ids have duplicates with DIFFERENT question text:\n")
    for id_, entries in differing.items():
        print(f"  {id_}:")
        for e in entries:
            print(f"    - {e['question'][:100]!r}")
        print()

    if not differing:
        print("None found -- safe to run scripts.dedupe_golden_set directly.")


if __name__ == "__main__":
    main()