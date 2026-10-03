import json,time
from eval.retrieval_dense import dense_search
from eval.retrieval_sparse import sparse_search
from eval.retrieval_hybrid import hybrid_search
from eval.retrieval_reranked import reranked_search
from eval.retrieval_metrics import reciprocal_rank,recall_at_k


FILENAME_MAP = {
    "aapl": "aapl-20250927.htm", "jpm": "jpm-20251231.htm", "xom": "xom-20251231.htm",
    "pfe": "pfe-20251231.htm", "wmt": "wmt-20250131.htm", "pld": "pld-20251231.htm",
    "cat": "cat-20251231.htm", "met": "met-20251231.htm", "ms": "ms-20251231.htm",
    "tsla": "tsla-20251231.htm",
}

MODES = {
    "dense": lambda q, c, f: dense_search(q, f),
    "sparse": lambda q, c, f: sparse_search(q, f),
    "hybrid": lambda q, c, f: hybrid_search(q, c, f),
    "hybrid_reranked": lambda q, c, f: reranked_search(q, c, f),
}

def run_matrix(golden_path="eval/golden_set_draft.json"):
    rows=json.load(open(golden_path))
    vector_rows = [r for r in rows if r["type"] == "vector_only"]   
    verified_vector_rows = [r for r in vector_rows if r.get("verified")]
    print("total vector_only rows:", len(vector_rows))
    print("verified vector_only rows:", len(verified_vector_rows))
    test_rows=[r for r in rows if r["type"]=="vector_only" and r.get("verified")]
    results={mode:{"mrr":[],"recall_5":[],"latencies":[]} for mode in MODES}

    for row in test_rows:
        company=row["company"]
        filename=FILENAME_MAP.get(company)
        if not filename:
            continue
        reference=row["reference_context"]

        for mode_name, mode_fn in MODES.items():
            start = time.time()
            retrieved = mode_fn(row["question"], company, filename)
            elapsed = time.time() - start

            results[mode_name]["mrr"].append(reciprocal_rank(retrieved, reference))
            results[mode_name]["recall_5"].append(recall_at_k(retrieved, reference, 5))
            results[mode_name]["latencies"].append(elapsed)
    print(f"{'Mode':<18}{'MRR':<8}{'Recall@5':<12}{'Avg Latency (s)'}")
    for mode_name, data in results.items():
        avg_mrr = sum(data["mrr"]) / len(data["mrr"])
        avg_recall = sum(data["recall_5"]) / len(data["recall_5"])
        avg_latency = sum(data["latencies"]) / len(data["latencies"])
        print(f"{mode_name:<18}{avg_mrr:<8.3f}{avg_recall:<12.3f}{avg_latency:.3f}")

if __name__ == "__main__":
    run_matrix()
        