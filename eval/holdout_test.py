# eval/holdout_test.py
from eval.retrieval_hybrid import hybrid_search
from eval.generation_variants import generate_answer
from eval.generation_metrics import verbatim_number_match, context_actually_supports_answer

# Deliberately NOT in the golden set, varied phrasing, mix of easy/hard,
# including one deliberately hard case (structural fact, likely to be
# under-retrieved, similar shape to the Tesla segment-count failure)
HOLDOUT_QUESTIONS = [
    {"company": "aapl", "filename": "aapl-20250927.htm", "question": "How many reportable segments does Apple have?", "expected": None},
    {"company": "jpm", "filename": "jpm-20251231.htm", "question": "What was JPMorgan's total revenue in 2025?", "expected": None},
    {"company": "xom", "filename": "xom-20251231.htm", "question": "Does ExxonMobil pay a dividend?", "expected": None},
    {"company": "pfe", "filename": "pfe-20251231.htm", "question": "What was Pfizer's effective tax rate in 2025?", "expected": None},
    {"company": "wmt", "filename": "wmt-20250131.htm", "question": "How many operating segments does Walmart report?", "expected": None},
    {"company": "cat", "filename": "cat-20251231.htm", "question": "What was Caterpillar's total revenue and sales for 2025?", "expected": None},
    {"company": "met", "filename": "met-20251231.htm", "question": "Who is MetLife's auditor?", "expected": None},
    {"company": "tsla", "filename": "tsla-20251231.htm", "question": "What products does Tesla's energy segment sell?", "expected": None},
]

def run_holdout():
    for item in HOLDOUT_QUESTIONS:
        print("=" * 80)
        print("Q:", item["question"])

        retrieved = hybrid_search(item["question"], item["company"], item["filename"], k=5)
        print(f"\n-- retrieved {len(retrieved)} chunks --")
        for r in retrieved:
            print(" ", r[:150].replace("\n", " "))

        answer = generate_answer(item["question"], "\n---\n".join(retrieved), "gpt-4o-mini", "strict_rules")
        print("\n-- generated answer --")
        print(answer)

        # if the answer contains a number, manually eyeball whether retrieval actually supported it
        print()

if __name__ == "__main__":
    run_holdout()