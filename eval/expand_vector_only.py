# eval/expand_vector_only.py
import json
from ingestion.embed_and_store import get_collection
from eval.build_golden_set import (
    COMPANIES, draft_narrative_qa, pick_narrative_chunk, _next_id
)

GOLDEN_PATH = "eval/golden_set_draft.json"

def expand_vector_only(additional_per_company: int = 3):
    rows = json.load(open(GOLDEN_PATH))

    # build the set of chunks already used across the EXISTING golden set,
    # so new generation never duplicates a question already tested
    already_used = set()
    for row in rows:
        if row["type"] == "vector_only" and isinstance(row.get("reference_context"), str):
            already_used.add(row["reference_context"])

    collection = get_collection(name="filing_chunks_mpnet")
    new_rows = []

    for c in COMPANIES:
        used_this_company = {u for u in already_used}  # start from full existing set
        added = 0
        attempts = 0
        while added < additional_per_company and attempts < additional_per_company * 3:
            attempts += 1
            chunk_text, meta = pick_narrative_chunk(c["pattern"], collection, used_this_company)
            if not chunk_text:
                break
            used_this_company.add(chunk_text[:500])  # match the truncation used when storing
            qa = draft_narrative_qa(c["name"], chunk_text)
            row = {
                "id": _next_id(f"{c['ticker']}_narr_exp"),
                "company": c["ticker"], "type": "vector_only",
                "question": qa["question"], "answer": qa["draft_answer"],
                "reference_context": chunk_text[:500],
                "verified": False,
            }
            new_rows.append(row)
            added += 1

    all_rows = rows + new_rows
    with open(GOLDEN_PATH, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"Added {len(new_rows)} new vector_only rows. Total rows: {len(all_rows)}")

if __name__ == "__main__":
    expand_vector_only()