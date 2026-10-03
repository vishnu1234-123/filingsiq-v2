# eval/holdout_vector_test.py
from eval.retrieval_hybrid import hybrid_search
from eval.generation_variants import generate_answer
from eval.generation_metrics import faithfulness_check

NARRATIVE_QUESTIONS = [
    {"company": "aapl", "filename": "aapl-20250927.htm", "question": "What intellectual property risks does Apple face?"},
    {"company": "jpm", "filename": "jpm-20251231.htm", "question": "How does JPMorgan describe its approach to cybersecurity governance?"},
    {"company": "xom", "filename": "xom-20251231.htm", "question": "What does ExxonMobil say about severe weather risks to its operations?"},
    {"company": "pfe", "filename": "pfe-20251231.htm", "question": "How does Pfizer describe competition in the biopharmaceutical industry?"},
    {"company": "wmt", "filename": "wmt-20250131.htm", "question": "What does Walmart say about supply chain risks?"},
    {"company": "cat", "filename": "cat-20251231.htm", "question": "How does Caterpillar describe its dealer network?"},
    {"company": "met", "filename": "met-20251231.htm", "question": "What does MetLife say about interest rate risk?"},
    {"company": "ms", "filename": "ms-20251231.htm", "question": "How does Morgan Stanley describe its wealth management business?"},
]

def run_narrative_holdout():
    for item in NARRATIVE_QUESTIONS:
        print("=" * 80)
        print("Q:", item["question"])

        retrieved = hybrid_search(item["question"], item["company"], item["filename"], k=5)
        context = "\n---\n".join(retrieved)
        print(f"\n-- retrieved {len(retrieved)} chunks --")
        for r in retrieved:
            print(" ", r[:150].replace("\n", " "))

        answer = generate_answer(item["question"], context, "gpt-4o-mini", "strict_rules")
        print("\n-- generated answer --")
        print(answer)

        is_faithful = faithfulness_check(answer, context)
        print(f"\n-- faithfulness check: {'PASS' if is_faithful else 'FAIL'} --")

if __name__ == "__main__":
    run_narrative_holdout()