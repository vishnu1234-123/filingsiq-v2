# eval/run_generation_matrix.py
import json
from eval.retrieval_hybrid import hybrid_search
from eval.generation_variants import generate_answer, MODELS, PROMPT_VARIANTS
from eval.generation_metrics import (
    verbatim_number_match, refusal_compliance,
    context_actually_supports_answer, faithfulness_check,
)

FILENAME_MAP = {
    "aapl": "aapl-20250927.htm", "jpm": "jpm-20251231.htm", "xom": "xom-20251231.htm",
    "pfe": "pfe-20251231.htm", "wmt": "wmt-20250131.htm", "pld": "pld-20251231.htm",
    "cat": "cat-20251231.htm", "met": "met-20251231.htm", "ms": "ms-20251231.htm",
    "tsla": "tsla-20251231.htm",
}

def run_matrix(golden_path="eval/golden_set_draft.json"):
    rows = json.load(open(golden_path))
    test_rows = [r for r in rows if r.get("verified") and r["type"] in
                 ("sql_only", "vector_only", "unanswerable_refusal")]

    results = {}
    for model_name in MODELS:
        for prompt_name in PROMPT_VARIANTS:
            key = f"{model_name} | {prompt_name}"
            scores = {
                "numeric_correct": [], "numeric_grounded": [],  # grounded = retrieval actually supported it
                "faithful": [],  # vector_only
                "refusal_correct": [],
            }

            for row in test_rows:
                company = row["company"].split(",")[0]
                filename = FILENAME_MAP.get(company)
                retrieved = []
                context = ""

                if filename and row["type"] in ("sql_only", "vector_only"):
                    retrieved = hybrid_search(row["question"], company, filename, k=5)
                    context = "\n---\n".join(retrieved)
                elif row["type"] == "sql_only":
                    context = row["reference_context"]

                answer = generate_answer(row["question"], context, model_name, prompt_name)

                if row["type"] == "sql_only":
                    match = verbatim_number_match(answer, row["answer"])
                    if match is not None:
                        scores["numeric_correct"].append(match)
                        # NEW: was the answer actually grounded in retrieval,
                        # or did the model already know it from pretraining?
                        grounded = context_actually_supports_answer(retrieved, row["answer"])
                        scores["numeric_grounded"].append(grounded)

                elif row["type"] == "vector_only":
                    # NEW: prose faithfulness, using the actual retrieved context
                    faithful = faithfulness_check(answer, context)
                    scores["faithful"].append(faithful)

                elif row["type"] == "unanswerable_refusal":
                    scores["refusal_correct"].append(
                        refusal_compliance(answer, row.get("refusal_flavor", ""))
                    )

            def avg(lst):
                return sum(lst) / len(lst) if lst else None

            results[key] = {
                "numeric_accuracy": avg(scores["numeric_correct"]),
                "numeric_grounded": avg(scores["numeric_grounded"]),
                "faithfulness": avg(scores["faithful"]),
                "refusal_compliance": avg(scores["refusal_correct"]),
            }

    header = f"{'Config':<28}{'Numeric':<10}{'Grounded':<11}{'Faithful':<11}{'Refusal'}"
    print(header)
    for key, r in results.items():
        def fmt(v):
            return f"{v:.3f}" if v is not None else "N/A"
        print(f"{key:<28}{fmt(r['numeric_accuracy']):<10}{fmt(r['numeric_grounded']):<11}"
              f"{fmt(r['faithfulness']):<11}{fmt(r['refusal_compliance'])}")

    return results

if __name__ == "__main__":
    run_matrix()