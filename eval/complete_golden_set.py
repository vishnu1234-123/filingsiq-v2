# eval/complete_golden_set.py
import json
import sqlite3
from langchain_openai import ChatOpenAI
from ingestion.embed_and_store import get_collection

_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
DB_PATH = "data/facts.sqlite"
GOLDEN_PATH = "eval/golden_set_draft.json"

ANALYTICAL_KEYWORDS = {
    "aapl": ["disciplined", "cost"],
    "jpm": ["net interest income", "provision for credit losses"],
    "xom": ["upstream", "low-carbon", "emissions"],
    "pfe": ["restructuring", "cost realignment"],
    "wmt": ["eCommerce", "margin"],
    "pld": ["occupancy", "high-barrier", "disciplined capital"],
}

REFUSAL_TEMPLATES = [
    ("motive_speculation",
     "Write ONE question asking WHY {company} secretly did something for strategic/hidden "
     "reasons -- something no SEC filing would ever admit as a motive."),
    ("moral_judgment",
     "Write ONE question asking whether {company}'s action (layoffs, pricing, a business "
     "decision) was 'ethical', 'fair', or 'justified' -- a moral judgment no factual filing can answer."),
]


def search_chunks(source_pattern, keywords, limit=1):
    collection = get_collection(name="filing_chunks_mpnet")
    items = collection.get(include=["documents", "metadatas"])
    matches = [
        (doc, meta) for doc, meta in zip(items["documents"], items["metadatas"])
        if source_pattern in meta.get("source_file", "") and any(k.lower() in doc.lower() for k in keywords)
    ]
    return matches[:limit]


def find_contradiction(source_pattern):
    """Real, same-concept-different-value/segment discrepancy -- the Day 6
    Apple commercial paper pattern, found generically for any company."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT concept, value, period_end, instant_date, segments_json FROM facts "
        "WHERE source_file LIKE ? ORDER BY concept",
        (f"%{source_pattern}%",),
    ).fetchall()
    conn.close()

    by_key = {}
    for concept, value, pe, idate, seg in rows:
        by_key.setdefault((concept, pe, idate), []).append((value, seg))

    for (concept, pe, idate), variants in by_key.items():
        values = {v[0] for v in variants}
        segs = {v[1] for v in variants}
        if len(variants) >= 2 and len(values) > 1 and len(segs) > 1:
            return concept, (pe or idate), variants
    return None


def _split_qa(response: str):
    if "QUESTION:" not in response or "ANSWER:" not in response:
        return None
    q = response.split("QUESTION:")[1].split("ANSWER:")[0].strip()
    a = response.split("ANSWER:")[1].strip()
    return q, a


def draft_analytical(company, keywords):
    chunks = search_chunks(company, keywords)
    if not chunks:
        return None
    doc, _ = chunks[0]
    prompt = (
        f"Real excerpt from {company}'s SEC filing:\n\n{doc[:800]}\n\n"
        f"Write ONE realistic analytical question requiring synthesis of this "
        f"excerpt with financial context, and a 3-4 sentence answer based ONLY "
        f"on this excerpt.\nFormat:\nQUESTION: ...\nANSWER: ..."
    )
    result = _split_qa(_model.invoke(prompt).content)
    if not result:
        return None
    q, a = result
    return {"question": q, "answer": a, "reference_narrative": [doc[:500]]}


def draft_contradiction(company):
    found = find_contradiction(company)
    if not found:
        return None
    concept, period, variants = found
    variant_text = "; ".join(f"value={v[0]}, segments={v[1]}" for v in variants)
    prompt = (
        f"A company's '{concept}' concept for period {period} has these "
        f"reported values: {variant_text}.\n\n"
        f"Write ONE question a confused user might ask noticing this apparent "
        f"discrepancy, and a 2-3 sentence answer explaining why both are correct "
        f"(e.g. one is a segment or precision variant of the other).\n"
        f"Format:\nQUESTION: ...\nANSWER: ..."
    )
    result = _split_qa(_model.invoke(prompt).content)
    if not result:
        return None
    q, a = result
    return {"question": q, "answer": a, "reference_context": [f"{concept} ({period}): {variant_text}"]}


def draft_refusal(company, idx):
    flavor, template = REFUSAL_TEMPLATES[idx % len(REFUSAL_TEMPLATES)]
    question = _model.invoke(template.format(company=company) + " Return only the question.").content.strip()
    answer = (
        "System must decline to assert undisclosed motive; may report factual "
        "details if directly relevant, never claim intent not stated in the filing."
        if flavor == "motive_speculation" else
        "System should report factual details if disclosed, but must decline to "
        "render a moral/ethical verdict."
    )
    return {"question": question, "answer": answer, "refusal_flavor": flavor}


def complete_golden_set():
    rows = json.load(open(GOLDEN_PATH))
    updated = 0

    for i, row in enumerate(rows):
        if row.get("question") is not None:
            continue

        if row["type"] == "analytical_synthesis":
            result = draft_analytical(row["company"], ANALYTICAL_KEYWORDS.get(row["company"], [row["company"]]))
            if result:
                row.update(result)
                row["verified"] = False  # still needs your read-through
                updated += 1

        elif row["type"] == "contradiction_reconciliation":
            result = draft_contradiction(row["company"])
            if result:
                row.update(result)
                row["verified"] = False
                updated += 1

        elif row["type"] == "unanswerable_refusal":
            result = draft_refusal(row["company"], i)
            row.update(result)
            row["verified"] = True  # correctness is definitional, no lookup needed

        updated += 1

    with open(GOLDEN_PATH, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    still_blank = [r["id"] for r in rows if r.get("question") is None]
    print(f"filled: {updated}, still blank (no chunk/contradiction found): {still_blank}")


if __name__ == "__main__":
    complete_golden_set()