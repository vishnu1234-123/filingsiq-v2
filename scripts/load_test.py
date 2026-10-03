"""
scripts/load_test.py

Synthetic load test for the router. Why this exists, and why it belongs
in the repo rather than being a one-off script someone runs by hand:

The gap between "the pipeline works" and "we know its operating
characteristics" is exactly where portfolio projects usually stop and
production systems don't. Before FastAPI/Docker/AWS sizing decisions --
how many workers, what autoscaling threshold, what rate limit per tenant
-- those decisions should be based on measured p95/p99 latency and cost
per query under realistic concurrency, not a guess. This script produces
those numbers against your own golden set, and is meant to be re-run
after every retrieval/prompt/model change so regressions show up as a
number changing, not a vibe.

Usage:
    python -m scripts.load_test --concurrency 10 --repeat 3
    python -m scripts.load_test --concurrency 1              # baseline, no contention
    python -m scripts.load_test --questions-file eval/golden_set_draft.json --n 20

Notes on what this does and doesn't simulate:
- Runs in-process against `answer_question()` directly -- it measures the
  router + retrieval + LLM latency, NOT network/API layer overhead (once
  FastAPI exists, a real HTTP load test, e.g. locust or k6 against the
  running service, is the thing to also run -- this script doesn't
  replace that, it's the layer below it).
- Concurrency is threads, not processes -- fine for measuring latency
  under concurrent LLM calls (I/O-bound), not a proxy for CPU-bound
  throughput limits.
- Uses your real MODEL and real retrieval by default, so it costs real
  money and hits real rate limits at high concurrency -- that's
  deliberate, since a load test against a mock tells you nothing about
  your actual OpenAI rate limit ceiling. Use --dry-run to sanity-check
  the harness itself.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_questions(path: str | None, n: int | None, question_key: str | None = None) -> list[str]:
    if path and Path(path).exists():
        data = json.loads(Path(path).read_text())
        candidate_keys = (question_key,) if question_key else ("question", "query", "q", "sub_question", "prompt")
        found_key = next((k for k in candidate_keys if data and k in data[0]), None)
        if found_key is None:
            actual_keys = sorted(data[0].keys()) if data else []
            raise ValueError(
                f"No question-like key found in {path!r}. Tried {candidate_keys}, "
                f"but the first row's keys are {actual_keys}. Pass --question-key "
                f"to name the right one, or fix load_questions() directly."
            )
        questions = [row[found_key] for row in data if found_key in row]
        if not questions:
            raise ValueError(f"Found key {found_key!r} but every row's value was empty in {path!r}.")
    else:
        questions = [
            "What was Apple's R&D spend in FY2025?",
            "How did JPMorgan's net income trend over the last two years?",
            "What does Pfizer's filing say about litigation risk?",
            "What was Walmart's total revenue in FY2025?",
            "Summarize Tesla's risk factors around supply chain.",
        ]
    if n:
        # cycle through if n exceeds the list, so --repeat-style load is
        # possible even off the small built-in sample
        questions = [questions[i % len(questions)] for i in range(n)]
    return questions


def run_one(question: str, dry_run: bool) -> dict:
    t0 = time.monotonic()
    try:
        if dry_run:
            time.sleep(0.05)  # harness smoke test, no real call
            result = {"blocked": False, "retry_count": 0, "faithfulness": 0.9,
                      "cost_summary": {"total_cost_usd": 0.0001}}
        else:
            from orchestration.router import answer_question
            result = answer_question(question)
        latency_ms = (time.monotonic() - t0) * 1000
        return {
            "question": question, "ok": True, "latency_ms": latency_ms,
            "blocked": result.get("blocked", False),
            "retry_count": result.get("retry_count", 0),
            "faithfulness": result.get("faithfulness"),
            "cost_usd": result.get("cost_summary", {}).get("total_cost_usd", 0.0),
        }
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        return {"question": question, "ok": False, "latency_ms": latency_ms, "error": str(exc)}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=1, help="how many times to loop over the question set")
    ap.add_argument("--questions-file", default="eval/golden_set_draft.json")
    ap.add_argument("--question-key", default=None, help="explicit key name if auto-detection fails")
    ap.add_argument("--n", type=int, default=None, help="cap/expand question count (cycles the set)")
    ap.add_argument("--dry-run", action="store_true", help="smoke-test the harness without calling the real router")
    ap.add_argument("--report-out", default="data/load_test_report.json")
    args = ap.parse_args()

    base_questions = load_questions(args.questions_file, args.n, args.question_key)
    questions = base_questions * args.repeat

    print(f"Load test: {len(questions)} requests, concurrency={args.concurrency}, dry_run={args.dry_run}")
    t_start = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, q, args.dry_run) for q in questions]
        for f in as_completed(futures):
            results.append(f.result())
    wall_time_s = time.monotonic() - t_start

    ok = [r for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]
    latencies = [r["latency_ms"] for r in ok]
    costs = [r["cost_usd"] for r in ok]
    retried = [r for r in ok if r["retry_count"] > 0]
    blocked = [r for r in ok if r["blocked"]]

    report = {
        "total_requests": len(results),
        "errors": len(errors),
        "error_rate_pct": round(100 * len(errors) / len(results), 2) if results else 0,
        "wall_time_s": round(wall_time_s, 2),
        "throughput_req_per_s": round(len(results) / wall_time_s, 3) if wall_time_s else 0,
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 1),
            "p95": round(percentile(latencies, 0.95), 1),
            "p99": round(percentile(latencies, 0.99), 1),
            "max": round(max(latencies), 1) if latencies else 0,
        },
        "cost_usd": {
            "total": round(sum(costs), 6),
            "avg_per_query": round(statistics.mean(costs), 6) if costs else 0,
        },
        "retry_rate_pct": round(100 * len(retried) / len(ok), 2) if ok else 0,
        "block_rate_pct": round(100 * len(blocked) / len(results), 2) if results else 0,
    }

    print(json.dumps(report, indent=2))
    if errors:
        print(f"\n{len(errors)} error(s), first: {errors[0]['error']}")

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()