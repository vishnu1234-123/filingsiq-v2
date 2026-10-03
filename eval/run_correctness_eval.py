"""
eval/run_correctness_eval.py

Runs the real pipeline against the golden set and scores actual
correctness -- not the same thing as the live faithfulness gate. A load
test shows what fraction of answers the guardrail *accepted*; this shows
what fraction were actually *right*, using each row's own ground truth.
The gap between those two numbers is real (see the Apple non-current-assets
and Tesla operating-segments cases from the last load test: both accepted
with faithfulness 1.0, both wrong).

Scoring differs by question type, because "correct" means different
things for a numeric SQL fact vs. a narrative paraphrase vs. a refusal:

  sql_only              -> verbatim_number_match(answer, row["answer"])
  vector_only           -> correctness_score(answer, row["answer"]) >= threshold
  analytical_synthesis  -> correctness_score(answer, row["answer"]) >= threshold
  contradiction_reconciliation -> correctness_score(answer, row["answer"]) >= threshold
  unanswerable_refusal  -> refusal_compliance(answer, row.get("refusal_flavor", ""))
  trend_multi_year      -> skipped (rows are self-documented "known-unbuilt")
  comparative_cross_company -> skipped (same, plus "company" is comma-joined
                                and nothing in entity_resolution splits it)

For sql_only rows specifically, also checks whether the golden concept tag
(parsed out of "reference_context") ever survived candidate_concepts()'s
list -- directly answers "did the reranking/matching pipeline even give
the LLM a chance to pick correctly," separate from whether it did.

Usage:
    python -m eval.run_correctness_eval
    python -m eval.run_correctness_eval --n 10                  # quick smoke test
    python -m eval.run_correctness_eval --types sql_only,vector_only
    python -m eval.run_correctness_eval --skip-known-unbuilt=false  # include them anyway
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

CORRECTNESS_PASS_THRESHOLD = 0.7
KNOWN_UNBUILT_TYPES = {"trend_multi_year", "comparative_cross_company"}


def load_golden_set(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of rows in {path!r}, got {type(data).__name__}")
    return data


def extract_golden_concept(reference_context) -> str | None:
    """Pulls 'us-gaap:SomeConcept' out of a reference_context string like
    'us-gaap:OtherAssetsNoncurrent = 83727000000.0 (2025-09-27)'. Returns
    None for rows where reference_context isn't that shape (list/dict/null)."""
    if not isinstance(reference_context, str):
        return None
    m = re.match(r"([\w.\-]+:[\w]+)\s*=", reference_context)
    return m.group(1) if m else None


def score_row(row: dict, answer: str) -> dict:
    """Returns {passed: bool, score: float|None, method: str, detail: str}."""
    from eval.generation_metrics import verbatim_number_match, correctness_score, refusal_compliance

    row_type = row["type"]
    if row_type == "sql_only":
        expected = row["answer"]
        matched = verbatim_number_match(answer, expected)
        return {"passed": bool(matched), "score": None, "method": "verbatim_number_match",
                "detail": f"expected={expected}"}

    if row_type in ("vector_only", "analytical_synthesis", "contradiction_reconciliation"):
        ground_truth = row["answer"]
        score = correctness_score(answer, ground_truth)
        return {"passed": score >= CORRECTNESS_PASS_THRESHOLD, "score": score,
                "method": "correctness_score", "detail": ""}

    if row_type == "unanswerable_refusal":
        complied = refusal_compliance(answer, row.get("refusal_flavor", ""))
        return {"passed": bool(complied), "score": None, "method": "refusal_compliance", "detail": ""}

    return {"passed": None, "score": None, "method": "unscored", "detail": f"no scorer for type {row_type!r}"}


def run_row(row: dict) -> dict:
    from orchestration.router import answer_question
    from orchestration.concept_resolver import candidate_concepts

    result = {
        "id": row["id"], "type": row["type"], "company": row["company"],
        "question": row["question"], "passed": None, "score": None,
        "method": None, "detail": "", "error": None,
        "golden_concept_in_candidates": None,
    }

    # Diagnostic: for sql_only rows, check whether the golden concept ever
    # made it into candidate_concepts() -- separate from whether the LLM
    # then picked it correctly. Answers the recall-vs-precision question
    # directly instead of inferring it from the final answer alone.
    if row["type"] == "sql_only":
        golden_concept = extract_golden_concept(row.get("reference_context"))
        if golden_concept:
            try:
                candidates = candidate_concepts(row["company"], row["question"])
                result["golden_concept_in_candidates"] = golden_concept in candidates
            except Exception as exc:
                result["golden_concept_in_candidates"] = None
                result["detail"] = f"candidate_concepts check failed: {exc}"

    try:
        # use_cache=False -- an eval MUST exercise the current code, never
        # a cached answer from before whatever fix is being tested. This
        # is not optional: confirmed in production that a full 20-row run
        # came back 100% cache hits and completely failed to reflect a
        # real fix that had just landed.
        pipeline_result = answer_question(row["question"], use_cache=False)
        # .get("answer", "") only falls back to "" when the key is MISSING
        # -- a blocked question leaves "answer" present but explicitly
        # None, so that call silently returned None here and crashed two
        # lines later on `"..." in answer`. `or ""` catches both cases.
        answer = pipeline_result.get("answer") or ""
        result["blocked"] = pipeline_result.get("blocked", False)
        result["retry_count"] = pipeline_result.get("retry_count", 0)
        result["faithfulness"] = pipeline_result.get("faithfulness")

        # The full story, not just pass/fail -- routes taken per
        # sub-question, whether web_fallback actually fired, cache
        # status, cost. Previously split across three scripts with no
        # single place to see all of it for one question at once.
        result["served_from_cache"] = pipeline_result.get("served_from_cache", False)
        result["cost_usd"] = pipeline_result.get("cost_summary", {}).get("total_cost_usd")
        sub_results = pipeline_result.get("sub_results") or []
        result["sub_questions"] = [
            # Field renamed from result_preview (was truncated [:200]) to
            # result -- a 200-char preview of met_narr_32's actual failure
            # looked like clean retrieval failure and wasn't; full text is
            # needed to tell "retrieval got nothing relevant" apart from
            # "retrieval got something relevant that synthesis ignored,"
            # which are different bugs needing different fixes.
            {"question": s.get("sub_question"), "route": s.get("route"),
             "origin": s.get("origin"),
             "status": s.get("status"), "result": s.get("result") or ""}
            for s in sub_results
        ]
        result["used_web_fallback"] = "[This answer used external web search" in answer or \
                                       "[Web search was unavailable" in answer

        if result["blocked"]:
            result["passed"] = False
            result["method"] = "blocked"
            result["detail"] = pipeline_result.get("block_reason", "")
            return result
        scored = score_row(row, answer)
        result.update(scored)
        result["answer"] = answer
    except Exception as exc:
        result["error"] = str(exc)
        result["passed"] = False
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-file", default="eval/golden_set_core.json",
                     help="defaults to the cleaned core set from prepare_golden_set.py; "
                          "falls back to golden_set_draft.json if core doesn't exist yet")
    ap.add_argument("--n", type=int, default=None, help="limit to first N eligible rows")
    ap.add_argument("--types", default=None, help="comma-separated list of types to include")
    ap.add_argument("--skip-known-unbuilt", default="true")
    ap.add_argument("--report-out", default="data/correctness_eval_report.json")
    ap.add_argument("--rows-out", default="data/correctness_eval_rows.jsonl")
    args = ap.parse_args()

    skip_unbuilt = args.skip_known_unbuilt.lower() != "false"
    type_filter = set(args.types.split(",")) if args.types else None

    questions_file = args.questions_file
    if not Path(questions_file).exists() and questions_file == "eval/golden_set_core.json":
        print(f"{questions_file} doesn't exist yet (run: python -m eval.prepare_golden_set). "
              f"Falling back to eval/golden_set_draft.json for now.")
        questions_file = "eval/golden_set_draft.json"

    all_rows = load_golden_set(questions_file)
    eligible = []
    skipped_unbuilt = 0
    for row in all_rows:
        if type_filter and row["type"] not in type_filter:
            continue
        if skip_unbuilt and row["type"] in KNOWN_UNBUILT_TYPES:
            skipped_unbuilt += 1
            continue
        eligible.append(row)
    if args.n:
        eligible = eligible[: args.n]

    print(f"Running correctness eval: {len(eligible)} rows "
          f"({skipped_unbuilt} skipped as known-unbuilt)")

    rows_out_path = Path(args.rows_out)
    rows_out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    t_start = time.monotonic()
    with open(rows_out_path, "w") as rows_f:
        for i, row in enumerate(eligible, 1):
            r = run_row(row)
            results.append(r)
            rows_f.write(json.dumps(r, default=str) + "\n")
            status = "PASS" if r["passed"] else ("ERROR" if r["error"] else "FAIL")
            print(f"[{i}/{len(eligible)}] {r['id']} ({r['type']}): {status}")

    wall_time_s = time.monotonic() - t_start

    by_type: dict[str, dict] = {}
    for r in results:
        t = r["type"]
        by_type.setdefault(t, {"n": 0, "passed": 0, "errors": 0, "scores": []})
        by_type[t]["n"] += 1
        if r["error"]:
            by_type[t]["errors"] += 1
        elif r["passed"]:
            by_type[t]["passed"] += 1
        if r["score"] is not None:
            by_type[t]["scores"].append(r["score"])

    sql_rows = [r for r in results if r["type"] == "sql_only" and r["golden_concept_in_candidates"] is not None]
    concept_recall = (
        sum(1 for r in sql_rows if r["golden_concept_in_candidates"]) / len(sql_rows) if sql_rows else None
    )

    total = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    total_errors = sum(1 for r in results if r["error"])

    report = {
        "total_rows": total,
        "skipped_known_unbuilt": skipped_unbuilt,
        "overall_accuracy_pct": round(100 * total_passed / total, 1) if total else 0,
        "error_rate_pct": round(100 * total_errors / total, 1) if total else 0,
        "wall_time_s": round(wall_time_s, 1),
        "golden_concept_candidate_recall_pct": (
            round(100 * concept_recall, 1) if concept_recall is not None else None
        ),
        "by_type": {
            t: {
                "n": d["n"],
                "accuracy_pct": round(100 * d["passed"] / d["n"], 1) if d["n"] else 0,
                "errors": d["errors"],
                "avg_score": round(sum(d["scores"]) / len(d["scores"]), 3) if d["scores"] else None,
            }
            for t, d in sorted(by_type.items())
        },
    }

    print("\n" + json.dumps(report, indent=2))
    report_out_path = Path(args.report_out)
    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    report_out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {report_out_path}\nPer-row detail: {rows_out_path}")


if __name__ == "__main__":
    main()