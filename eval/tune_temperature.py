# eval/tune_temperature.py
import json
from langchain_openai import ChatOpenAI
from eval.retrieval_hybrid import hybrid_search
from eval.generation_variants import PROMPT_VARIANTS
from eval.generation_metrics import faithfulness_score, correctness_score

FILENAME_MAP = {
    "aapl": "aapl-20250927.htm", "jpm": "jpm-20251231.htm", "xom": "xom-20251231.htm",
    "pfe": "pfe-20251231.htm", "wmt": "wmt-20250131.htm", "pld": "pld-20251231.htm",
    "cat": "cat-20251231.htm", "met": "met-20251231.htm", "ms": "ms-20251231.htm",
    "tsla": "tsla-20251231.htm",
}

def run_temperature_sweep(temps=(0.0, 0.3, 0.7), sample_size=20):
    rows = json.load(open("eval/golden_set_draft.json"))
    test_rows = [r for r in rows if r.get("verified") and r["type"] == "vector_only"][:sample_size]

    for temp in temps:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=temp)
        faith_scores, correct_scores = [], []

        for row in test_rows:
            filename = FILENAME_MAP.get(row["company"])
            retrieved = hybrid_search(row["question"], row["company"], filename, k=5)
            context = "\n---\n".join(retrieved)
            prompt = PROMPT_VARIANTS["strict_rules"](row["question"], context)
            answer = model.invoke(prompt).content

            faith_scores.append(faithfulness_score(answer, context))
            correct_scores.append(correctness_score(answer, row["answer"]))

        avg_faith = sum(faith_scores) / len(faith_scores)
        avg_correct = sum(correct_scores) / len(correct_scores)
        print(f"temperature={temp}: faithfulness={avg_faith:.3f}  correctness={avg_correct:.3f}")

if __name__ == "__main__":
    run_temperature_sweep()