"""
scripts/view_trace.py

Reconstructs a readable per-question narrative from data/audit_log.jsonl,
instead of manually cross-referencing trace_id across scattered JSON
lines by eye. Every event in the audit log carries a trace_id; this just
groups by that and prints the story in order.

Usage:
    python -m scripts.view_trace --summary              # one line per trace, newest first
    python -m scripts.view_trace --last 5                # full narrative for the 5 most recent traces
    python -m scripts.view_trace --trace-id <id>          # full narrative for one specific trace
    python -m scripts.view_trace --contains "smartwatch"  # traces whose question matches a substring
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_LOG_PATH = "data/audit_log.jsonl"


def load_events(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} doesn't exist yet -- run a query through answer_question() first.")
    events = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def group_by_trace(events: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        grouped[e.get("trace_id", "unknown")].append(e)
    return grouped


def trace_question(trace_events: list[dict]) -> str:
    """Best-known question text for a trace, preferring the top-level
    query_received event (added specifically so this works even for
    successful traces), falling back to whatever's available for older
    log lines that predate it."""
    for e in trace_events:
        if e["event"] == "query_received":
            return e["question"]
    for e in trace_events:
        if e["event"] == "web_fallback_triggered":
            return e["question"]
    for e in trace_events:
        if e["event"] == "subquestion_routed":
            return e["sub_question"] + "  (sub-question only -- original question not logged for this trace)"
    return "(unknown -- no question-bearing event in this trace)"


def trace_final_decision(trace_events: list[dict]) -> str:
    complete = [e for e in trace_events if e["event"] == "trace_complete"]
    if complete:
        c = complete[-1]
        if c.get("blocked"):
            return "BLOCKED"
        return f"answered (faithfulness={c.get('faithfulness')}, retries={c.get('retry_count')})"
    blocked = [e for e in trace_events if e["event"] == "blocked"]
    if blocked:
        return f"BLOCKED ({blocked[0].get('reason')})"
    web = [e for e in trace_events if e["event"] == "web_fallback_triggered"]
    if web:
        return "web_fallback (older log -- predates trace_complete)"
    return "incomplete / in progress"


def print_narrative(trace_id: str, trace_events: list[dict]) -> None:
    print("=" * 72)
    print(f"TRACE {trace_id}")
    print("=" * 72)
    print(f"Q: {trace_question(trace_events)}\n")

    if any(e["event"] == "blocked" for e in trace_events):
        b = next(e for e in trace_events if e["event"] == "blocked")
        print(f"  BLOCKED at input: {b.get('reason')}\n")
        return

    # Group subquestion_routed + faithfulness_computed + guardrail_decision
    # into attempts, in the order they actually happened.
    attempt = 0
    for e in trace_events:
        if e["event"] == "subquestion_routed":
            print(f"  [attempt {attempt}] sub-question: {e['sub_question']!r} (company={e.get('company')!r})")
            print(f"      route={e['route']!r} status={e.get('status', '?')!r}")
            print(f"      -> {e['result_preview'][:150]}")
        elif e["event"] == "subquestion_retried":
            print(f"  [attempt {attempt}] RETRIED sub-question: {e['sub_question']!r} (company={e.get('company')!r})")
            print(f"      route={e['route']!r} status={e.get('status', '?')!r}")
            print(f"      -> {e['result_preview'][:150]}")
        elif e["event"] == "faithfulness_computed":
            print(f"      faithfulness={e['score']} (context_len={e.get('context_len')})")
        elif e["event"] == "guardrail_decision":
            print(f"      decision={e['decision']!r} (failure_type={e.get('failure_type', '?')!r})")
            if e["decision"] in ("retry_retrieval", "retry_synthesis"):
                attempt += 1
            print()
        elif e["event"] == "web_fallback_triggered":
            print("  >>> WEB FALLBACK TRIGGERED <<<")
        elif e["event"] == "web_search_failed":
            print(f"      (web search failed: provider={e.get('provider')}, error={e.get('error')})")
        elif e["event"] == "mpnet_rerank_failed":
            print(f"      (mpnet rerank fell back to {e.get('fallback')}: {e.get('error')})")
        elif e["event"] == "trace_complete":
            print(f"  FINAL: {trace_final_decision(trace_events)}")
            if e.get("answer_preview"):
                print(f"  answer: {e['answer_preview']}")
            if e.get("cost_usd") is not None:
                print(f"  cost: ${e['cost_usd']:.6f}")
    print()


def print_summary(grouped: dict[str, list[dict]]) -> None:
    print(f"{'trace_id':<14} {'decision':<45} question")
    print("-" * 100)
    for trace_id, events in grouped.items():
        q = trace_question(events)
        decision = trace_final_decision(events)
        q_short = (q[:55] + "...") if len(q) > 58 else q
        print(f"{trace_id:<14} {decision:<45} {q_short}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    ap.add_argument("--trace-id", default=None)
    ap.add_argument("--last", type=int, default=None, help="show full narrative for the N most recent traces")
    ap.add_argument("--summary", action="store_true", help="one line per trace instead of full narrative")
    ap.add_argument("--contains", default=None, help="only traces whose question contains this substring")
    args = ap.parse_args()

    events = load_events(args.log_path)
    grouped = group_by_trace(events)

    # Preserve first-seen order per trace (already true from the JSONL's
    # natural append order), and order traces themselves by first event.
    trace_order = list(dict.fromkeys(e.get("trace_id", "unknown") for e in events))

    if args.contains:
        trace_order = [t for t in trace_order if args.contains.lower() in trace_question(grouped[t]).lower()]

    if args.trace_id:
        if args.trace_id not in grouped:
            print(f"No events found for trace_id={args.trace_id!r}")
            return
        print_narrative(args.trace_id, grouped[args.trace_id])
        return

    if args.summary:
        # newest first for scanning
        ordered = {t: grouped[t] for t in reversed(trace_order)}
        print_summary(ordered)
        return

    n = args.last or 5
    for trace_id in reversed(trace_order[-n:]):
        print_narrative(trace_id, grouped[trace_id])


if __name__ == "__main__":
    main()