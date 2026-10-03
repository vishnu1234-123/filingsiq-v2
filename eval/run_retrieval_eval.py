"""
eval/run_retrieval_eval.py

Isolates retrieval recall from everything downstream of it (synthesis,
faithfulness, the guardrail). For each vector_only golden-set row, checks
whether the known-correct chunk (reference_context -- several rows are
annotated as verified verbatim against the stored chunk) actually comes
back from run_vector(), at more than one k value. This answers exactly
the question that came up debugging the Apple R&D case: was that a
"never retrievable regardless of k" gap, or a "retrievable with a wider
k / better reranking" gap? Now it's answered for every vector question
in the golden set at once, not one at a time by hand.

Match strategy: checks whether the first ~120 characters of
reference_context appear in the retrieved text, not the whole string.
Reference_context is sometimes truncated mid-sentence in the golden set
itself, and even a real chunk hit can have slightly different boundaries
than the stored reference -- a short, distinctive prefix is a more
robust fingerprint than requiring an exact full-string match.

Usage:
    python -m eval.run_retrieval_eval
    python -m eval.run_retrieval_eval --k-values 5,10,20
    python -m eval.run_retrieval_eval --n 10          # quick smoke test
    python -m eval.run_retrieval_eval --hyde           # also test with HyDE on
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MATCH_PREFIX_LEN = 120


def load_vector_rows(path: str, n: int | None) -> list[dict]:
    data = json.loads(Path(path).read_text())
    rows = [r for r in data if r.get("type") == "vector_only" and isinstance(r.get("reference_context"), str)]
    if n:
        rows = rows[:n]
    return rows


import re
import unicodedata


def _normalize(text: str) -> str:
    """
    Collapses exactly the kind of differences that are formatting noise,
    not retrieval failure: curly vs straight quotes (confirmed present in
    the golden set itself -- "Company\u2019s" uses a curly apostrophe),
    other unicode punctuation variants, and whitespace differences from
    chunking/extraction. Without this, a chunk-boundary or quote-style
    difference between when the golden set was authored and now would
    show up as a false "never_retrieved" that has nothing to do with
    whether retrieval actually found the right content.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def is_retrieved(reference_context: str, retrieved_text: str) -> bool:
    needle = _normalize(reference_context)[:MATCH_PREFIX_LEN]
    return needle in _normalize(retrieved_text)


def run_row(row: dict, k_values: list[int], use_hyde: bool) -> dict:
    from orchestration.router import run_vector

    result = {
        "id": row["id"], "company": row["company"], "question": row["question"],
        "reference_prefix": row["reference_context"].strip()[:MATCH_PREFIX_LEN],
        "found_at_k": {}, "error": None,
        "smallest_k_retrieved_preview": None,  # what actually came back at the smallest k, when it wasn't a hit
    }
    try:
        smallest_k = min(k_values)
        for k in k_values:
            text, status = run_vector(row["company"], row["question"], k=k, use_hyde=use_hyde)
            hit = is_retrieved(row["reference_context"], text)
            result["found_at_k"][k] = hit
            if k == smallest_k and not hit:
                # Captures what WAS retrieved instead, so "recall improves
                # at wider k" doesn't stay ambiguous between "the k=5
                # chunks were relevant but insufficient" (needs more
                # chunks) and "the k=5 chunks were off-topic noise
                # outranking the real answer" (needs better ranking, not
                # more chunks) -- two different problems, two different
                # fixes, previously indistinguishable from found_at_k alone.
                result["smallest_k_retrieved_preview"] = text[:500]
    except Exception as exc:
        result["error"] = str(exc)
    return result


def classify(result: dict, k_values: list[int]) -> str:
    if result["error"]:
        return "error"
    found = result["found_at_k"]
    smallest_k, largest_k = min(k_values), max(k_values)
    if found.get(smallest_k):
        return "retrieved_at_smallest_k"          # working fine, no issue
    if any(found.get(k) for k in k_values):
        return "needs_wider_k_or_reranking"        # retrievable, just not at current k
    return "never_retrieved"                        # genuine gap -- widening k won't help


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-file", default="eval/golden_set_draft.json")
    ap.add_argument("--k-values", default="5,15")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--hyde", action="store_true")
    ap.add_argument("--report-out", default="data/retrieval_eval_report.json")
    ap.add_argument("--rows-out", default="data/retrieval_eval_rows.jsonl")
    args = ap.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]
    rows = load_vector_rows(args.questions_file, args.n)
    print(f"Testing retrieval recall for {len(rows)} vector_only questions at k={k_values}")

    rows_out_path = Path(args.rows_out)
    rows_out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.monotonic()
    with open(rows_out_path, "w") as f:
        for i, row in enumerate(rows, 1):
            r = run_row(row, k_values, args.hyde)
            r["classification"] = classify(r, k_values)
            results.append(r)
            f.write(json.dumps(r, default=str) + "\n")
            print(f"[{i}/{len(rows)}] {r['id']}: {r['classification']}")

    wall_time_s = time.monotonic() - t0

    counts = {}
    for r in results:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1

    total = len(results)
    report = {
        "total_questions": total,
        "wall_time_s": round(wall_time_s, 1),
        "k_values_tested": k_values,
        "breakdown": counts,
        "recall_at_smallest_k_pct": round(100 * counts.get("retrieved_at_smallest_k", 0) / total, 1) if total else 0,
        "recall_at_largest_k_pct": round(
            100 * (counts.get("retrieved_at_smallest_k", 0) + counts.get("needs_wider_k_or_reranking", 0)) / total, 1
        ) if total else 0,
        "never_retrieved_pct": round(100 * counts.get("never_retrieved", 0) / total, 1) if total else 0,
    }

    print("\n" + json.dumps(report, indent=2))
    print("\nnever_retrieved rows (widening k/reranking will NOT fix these -- check chunking/ingestion instead):")
    for r in results:
        if r["classification"] == "never_retrieved":
            print(f"  {r['id']}: {r['question']!r}")

    print(f"\nneeds_wider_k_or_reranking rows -- what came back at k={min(k_values)} instead of the target "
          f"(on-topic-but-insufficient means 'add more chunks'; off-topic means 'fix ranking, not k'):")
    for r in results:
        if r["classification"] == "needs_wider_k_or_reranking":
            preview = (r.get("smallest_k_retrieved_preview") or "")[:150]
            print(f"  {r['id']}: {r['question']!r}")
            print(f"      got instead: {preview!r}")

    Path(args.report_out).write_text(json.dumps(report, indent=2))
    print(f"\nReport: {args.report_out}\nPer-row detail: {rows_out_path}")


if __name__ == "__main__":
    main()