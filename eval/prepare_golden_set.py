"""
eval/prepare_golden_set.py

Splits golden_set_draft.json into three purpose-built files instead of
one undifferentiated pile:

- golden_set_core.json: in-scope, non-duplicate rows. What correctness/
  routing accuracy should actually be measured against.
- golden_set_paraphrase_robustness.json: near-duplicate rows (same fact,
  reworded), each tagged with which core row it paraphrases. A real,
  separate question -- "does rewording change routing or the answer" --
  not accuracy noise.
- golden_set_future_capabilities.json: rows tagged "known-unbuilt" in
  their own notes (currently trend_multi_year, comparative_cross_company).
  Kept for later, excluded from current scoring since they fail for a
  reason that has nothing to do with pipeline quality.

Duplicate detection uses text similarity (difflib), not ID naming
patterns like "_exp_" -- so it keeps working if new near-duplicate rows
get added later without following that convention. Two vector_only rows
for the SAME company with question-text similarity above the threshold
are grouped; the first one marked "verified": true (or just the first,
if none are) becomes canonical and goes to core, the rest go to
paraphrase_robustness tagged with base_id pointing at the canonical one.

Usage:
    python -m eval.prepare_golden_set
    python -m eval.prepare_golden_set --similarity-threshold 0.55
    python -m eval.prepare_golden_set --input eval/golden_set_draft.json
"""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

UNBUILT_NOTE_MARKER = "known-unbuilt"
DEFAULT_SIMILARITY_THRESHOLD = 0.60


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_duplicate_groups(rows: list[dict], threshold: float) -> list[list[dict]]:
    """
    Groups rows (within the same company) whose questions are similar
    enough to be considered the same underlying fact reworded. Only
    compares rows of type vector_only -- sql_only/compute questions that
    happen to share phrasing usually aren't true duplicates the way
    reworded narrative questions are (different years/concepts read as
    similar text but are NOT the same question).
    """
    candidates = [r for r in rows if r.get("type") == "vector_only"]
    used = set()
    groups = []
    for i, r1 in enumerate(candidates):
        if r1["id"] in used:
            continue
        group = [r1]
        used.add(r1["id"])
        for r2 in candidates[i + 1:]:
            if r2["id"] in used or r2.get("company") != r1.get("company"):
                continue
            if similarity(r1["question"], r2["question"]) >= threshold:
                group.append(r2)
                used.add(r2["id"])
        if len(group) > 1:
            groups.append(group)
    return groups


def pick_canonical(group: list[dict]) -> dict:
    verified = [r for r in group if r.get("verified") is True]
    return verified[0] if verified else group[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="eval/golden_set_draft.json")
    ap.add_argument("--output-dir", default="eval")
    ap.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    args = ap.parse_args()

    rows = json.loads(Path(args.input).read_text())
    print(f"Loaded {len(rows)} rows from {args.input}")

    # 1. Future capabilities: tagged known-unbuilt in their own notes
    future = [r for r in rows if UNBUILT_NOTE_MARKER in (r.get("note") or "")]
    future_ids = {r["id"] for r in future}
    remaining = [r for r in rows if r["id"] not in future_ids]
    print(f"Future capabilities (known-unbuilt): {len(future)} rows")

    # 2. Duplicate detection among what's left
    groups = find_duplicate_groups(remaining, args.similarity_threshold)
    paraphrase_ids = set()
    paraphrase_rows = []
    for group in groups:
        canonical = pick_canonical(group)
        for r in group:
            if r["id"] != canonical["id"]:
                paraphrase_ids.add(r["id"])
                paraphrase_rows.append({**r, "base_id": canonical["id"]})
    print(f"Duplicate groups found: {len(groups)}, "
          f"{len(paraphrase_ids)} rows classified as paraphrases of a canonical row")

    core = [r for r in remaining if r["id"] not in paraphrase_ids]
    print(f"Core (in-scope, non-duplicate): {len(core)} rows")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "golden_set_core.json").write_text(json.dumps(core, indent=2))
    (out_dir / "golden_set_paraphrase_robustness.json").write_text(json.dumps(paraphrase_rows, indent=2))
    (out_dir / "golden_set_future_capabilities.json").write_text(json.dumps(future, indent=2))

    print(f"\nWrote:")
    print(f"  {out_dir / 'golden_set_core.json'} ({len(core)} rows)")
    print(f"  {out_dir / 'golden_set_paraphrase_robustness.json'} ({len(paraphrase_rows)} rows)")
    print(f"  {out_dir / 'golden_set_future_capabilities.json'} ({len(future)} rows)")

    by_type = {}
    for r in core:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    print(f"\nCore breakdown by type: {json.dumps(by_type, indent=2)}")

    print("\nSpot-check these duplicate groupings -- similarity detection is a heuristic, not ground truth:")
    for group in groups:
        canonical = pick_canonical(group)
        others = [r["id"] for r in group if r["id"] != canonical["id"]]
        print(f"  canonical={canonical['id']!r} <- paraphrases={others}")


if __name__ == "__main__":
    main()