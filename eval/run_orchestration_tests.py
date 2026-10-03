
import json
from pathlib import Path

from orchestration.router import answer_question
from eval.orchestration_test_set import ORCHESTRATION_TESTS

REPORT_PATH = "data/orchestration_test_report.json"
ROWS_PATH = "data/orchestration_test_rows.jsonl"


def run_orchestration_tests():
    results = []
    Path(ROWS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(ROWS_PATH, "w") as rows_f:
        for t in ORCHESTRATION_TESTS:
            r = answer_question(t["question"])
            passed = True
            notes = []

            if "expect_blocked" in t and r["blocked"] != t["expect_blocked"]:
                passed = False
                notes.append(f"expected blocked={t['expect_blocked']}, got {r['blocked']}")

            if "expect_reason" in t and r.get("block_reason") != t["expect_reason"]:
                passed = False
                notes.append(f"expected reason={t['expect_reason']}, got {r.get('block_reason')}")

            # Same fix as expect_route_type: a cache hit means
            # dispatch_subquestions never ran at all, so sub_results is
            # None -- that's not evidence decomposition failed, it's
            # evidence decomposition was never attempted this run.
            # Missing this exclusion here (while having it on
            # expect_route_type) caused mix_1/mix_2 to fail in production
            # purely because Apple R&D questions had been cached from
            # earlier testing, not because decomposition was broken.
            if "expect_subq_count_gte" in t and not r["blocked"] and not r.get("served_from_cache"):
                actual = len(r.get("sub_results") or [])
                if actual < t["expect_subq_count_gte"]:
                    passed = False
                    notes.append(f"expected >= {t['expect_subq_count_gte']} sub-questions, got {actual}")

            if "expect_route_type" in t and not r["blocked"] and not r.get("served_from_cache"):
                sub_results = r.get("sub_results") or []
                actual_routes = [s.get("route") for s in sub_results]
                if t["expect_route_type"] not in actual_routes:
                    passed = False
                    notes.append(f"expected route {t['expect_route_type']!r} among sub-questions, got {actual_routes}")

            if "expect_cache_hit" in t:
                actual_hit = bool(r.get("served_from_cache"))
                if actual_hit != t["expect_cache_hit"]:
                    passed = False
                    notes.append(f"expected served_from_cache={t['expect_cache_hit']}, got {actual_hit}")

            row = {
                "id": t["id"], "question": t["question"], "passed": passed, "notes": notes,
                "answer": r.get("answer"), "blocked": r.get("blocked"),
                "served_from_cache": r.get("served_from_cache"), "cost_usd": r.get("cost_summary", {}).get("total_cost_usd"),
            }
            results.append(row)
            rows_f.write(json.dumps(row, default=str) + "\n")
            print(f"[{'PASS' if passed else 'FAIL'}] {t['id']}: {t['question'][:60]!r} {notes}")

    total = len(results)
    passed = sum(r["passed"] for r in results)
    report = {
        "total": total, "passed": passed, "failed": total - passed,
        "failed_ids": [r["id"] for r in results if not r["passed"]],
    }
    Path(REPORT_PATH).write_text(json.dumps(report, indent=2))
    print(f"\n{passed}/{total} orchestration tests passed")
    print(f"Report: {REPORT_PATH}\nPer-row detail: {ROWS_PATH}")
    return results


if __name__ == "__main__":
    run_orchestration_tests()