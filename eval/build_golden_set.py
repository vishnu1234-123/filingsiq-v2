# eval/build_golden_set.py
import json
import sqlite3
import random
from langchain_openai import ChatOpenAI
from ingestion.embed_and_store import get_collection

_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
DB_PATH = "data/facts.sqlite"

COMPANIES = [
    {"ticker": "aapl", "pattern": "aapl", "name": "Apple", "sector": "tech"},
    {"ticker": "jpm", "pattern": "jpm", "name": "JPMorgan", "sector": "banking"},
    {"ticker": "xom", "pattern": "xom", "name": "ExxonMobil", "sector": "energy"},
    {"ticker": "pfe", "pattern": "pfe", "name": "Pfizer", "sector": "pharma"},
    {"ticker": "wmt", "pattern": "wmt", "name": "Walmart", "sector": "retail"},
    {"ticker": "pld", "pattern": "pld", "name": "Prologis", "sector": "reit"},
    {"ticker": "cat", "pattern": "cat", "name": "Caterpillar", "sector": "industrials"},
    {"ticker": "met", "pattern": "met", "name": "MetLife", "sector": "insurance"},
    {"ticker": "ms", "pattern": "ms", "name": "Morgan Stanley", "sector": "banking"},
    {"ticker": "tsla", "pattern": "tsla", "name": "Tesla", "sector": "auto"},
]

SUBSTANTIVE_HEADINGS = ["risk factors", "management's discussion", "cybersecurity", "legal proceedings", "business"]
_row_counter = [0]

def _next_id(prefix: str) -> str:
    _row_counter[0] += 1
    return f"{prefix}_{_row_counter[0]}"


def _company_name_present(question: str, name: str) -> bool:
    return name.lower() in question.lower()


def _enforce_company_name(question: str, name: str) -> str:
    """If the LLM's question drifted and dropped the company name
    entirely, prepend it rather than silently shipping an ambiguous
    question -- there's no conversational memory to fall back on."""
    if _company_name_present(question, name):
        return question
    return f"For {name}: {question}"


# ---- data access ----

def pick_numeric_facts(pattern: str, limit: int = 2) -> list[tuple]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT DISTINCT concept, value, period_end, instant_date FROM facts "
        "WHERE source_file LIKE ? AND segments_json = '[]' ORDER BY RANDOM() LIMIT ?",
        (f"%{pattern}%", limit),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def pick_concept_across_years(pattern: str, concept: str) -> list[tuple]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT value, period_end, instant_date FROM facts "
        "WHERE source_file LIKE ? AND concept = ? AND segments_json = '[]' "
        "ORDER BY COALESCE(period_end, instant_date)",
        (f"%{pattern}%", concept),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def find_common_concept(pattern_a: str, pattern_b: str) -> str | None:
    conn = sqlite3.connect(DB_PATH)
    a = {r[0] for r in conn.execute("SELECT DISTINCT concept FROM facts WHERE source_file LIKE ?", (f"%{pattern_a}%",))}
    b = {r[0] for r in conn.execute("SELECT DISTINCT concept FROM facts WHERE source_file LIKE ?", (f"%{pattern_b}%",))}
    conn.close()
    common = [c for c in (a & b) if c.startswith("us-gaap:")]
    return random.choice(common) if common else None


def pick_narrative_chunk(pattern: str, collection, exclude_texts: set):
    """Now excludes already-used chunks so the same company doesn't
    generate two near-duplicate questions off the identical excerpt."""
    all_items = collection.get(include=["documents", "metadatas"])
    candidates = [
        (d, m) for d, m in zip(all_items["documents"], all_items["metadatas"])
        if pattern in m.get("source_file", "") and m.get("type") == "text" and len(d) > 300
        and any(h in (m.get("heading") or "").lower() for h in SUBSTANTIVE_HEADINGS)
        and d not in exclude_texts
        and "us-gaap:" not in d[:100]  # reject raw XBRL-dump chunks upfront
    ]
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: sum(c.isdigit() for c in x[0]) / len(x[0]))
    return candidates[0]


# ---- LLM generation ----

def _style_instruction(style: str) -> str:
    if style == "scenario":
        return ("Frame it as a realistic scenario or context a person would give before "
                "asking their question, then ask naturally within it.")
    return "Write it as a direct, straightforward question."


def generate_numeric_question(company: str, concept: str, value, period, style="direct") -> str:
    prompt = (
        f"{company}'s '{concept}' concept has the value {value} for the specific period "
        f"{period}.\n\nWrite ONE question this fact answers. You MUST name '{company}' "
        f"explicitly in the question, and you MUST make the specific time period "
        f"unambiguous (e.g. name the exact year or date) -- do not use vague phrasing "
        f"like 'currently' or 'these investments' that could apply to multiple periods. "
        f"{_style_instruction(style)} Use plain language, not the raw concept name. "
        f"Don't include the value. Return only the question."
    )
    q = _model.invoke(prompt).content.strip()
    return _enforce_company_name(q, company)


def draft_narrative_qa(company: str, chunk_text: str) -> dict:
    prompt = (
        f"Real excerpt from {company}'s SEC filing:\n\n{chunk_text}\n\n"
        f"Write ONE realistic question this excerpt answers, and a 2-3 sentence "
        f"answer based ONLY on this excerpt. You MUST name '{company}' explicitly "
        f"in the question. Format:\nQUESTION: ...\nANSWER: ..."
    )
    response = _model.invoke(prompt).content
    if "QUESTION:" not in response or "ANSWER:" not in response:
        raise ValueError(f"Unexpected LLM response format:\n{response}")
    q = response.split("QUESTION:")[1].split("ANSWER:")[0].strip()
    a = response.split("ANSWER:")[1].strip()
    return {"question": _enforce_company_name(q, company), "draft_answer": a}


def generate_trend_question(company: str, concept: str, style="direct") -> str:
    prompt = (
        f"A financial concept named '{concept}' represents some real financial "
        f"metric for {company}. Infer what it means in plain English from the "
        f"concept name, then write ONE natural question asking how that metric "
        f"has changed over the past few years. You MUST name '{company}' "
        f"explicitly. Do NOT include the raw concept name, 'us-gaap:', or any "
        f"namespaced tag text in your question -- translate it fully into plain "
        f"language a real person would use. {_style_instruction(style)}"
    )
    q = _model.invoke(prompt).content.strip()
    return _enforce_company_name(q, company)


def generate_comparative_question(company_a: str, company_b: str, concept: str) -> str:
    prompt = (
        f"A financial concept named '{concept}' represents some real financial "
        f"metric. Infer what it means in plain English, then write ONE question "
        f"comparing {company_a} and {company_b} on that metric. You MUST name "
        f"both '{company_a}' and '{company_b}' explicitly. Do NOT include the "
        f"raw concept name or any namespaced tag text -- translate it fully."
        f"Return ONLY the QUESTION."
    )
    return _model.invoke(prompt).content.strip()


# ---- builders ----

def build_sql_only(n_per_company=2):
    rows = []
    for c in COMPANIES:
        for concept, value, period_end, instant_date in pick_numeric_facts(c["pattern"], n_per_company):
            period = period_end or instant_date
            style = random.choice(["direct", "scenario"])
            q = generate_numeric_question(c["name"], concept, value, period, style)
            rows.append({
                "id": _next_id(f"{c['ticker']}_sql"), "company": c["ticker"], "type": "sql_only",
                "question": q, "answer": value,
                "reference_context": f"{concept} = {value} ({period})",
                "verified": False,
            })
    return rows


def build_vector_only(n_per_company=2):
    rows = []
    collection = get_collection(name="filing_chunks_mpnet")
    for c in COMPANIES:
        used = set()
        for _ in range(n_per_company):
            chunk_text, meta = pick_narrative_chunk(c["pattern"], collection, used)
            if not chunk_text:
                continue
            used.add(chunk_text)
            qa = draft_narrative_qa(c["name"], chunk_text)
            rows.append({
                "id": _next_id(f"{c['ticker']}_narr"), "company": c["ticker"], "type": "vector_only",
                "question": qa["question"], "answer": qa["draft_answer"],
                "reference_context": chunk_text[:500],
                "verified": False,
            })
    return rows


def build_trend_multi_year(n_per_company=1):
    rows = []
    for c in COMPANIES:
        facts = pick_numeric_facts(c["pattern"], 1)
        if not facts:
            continue
        concept = facts[0][0]
        history = pick_concept_across_years(c["pattern"], concept)
        if len(history) < 2:
            continue
        q = generate_trend_question(c["name"], concept, random.choice(["direct", "scenario"]))
        rows.append({
            "id": _next_id(f"{c['ticker']}_trend"), "company": c["ticker"], "type": "trend_multi_year",
            "question": q, "answer": None,
            "reference_context": [f"{v} ({p or i})" for v, p, i in history],
            "verified": False, "note": "known-unbuilt: requires compute tool",
        })
    return rows


def build_comparative(n_pairs=5):
    rows = []
    pairs = [(COMPANIES[i], COMPANIES[i + 1]) for i in range(0, min(n_pairs * 2, len(COMPANIES) - 1), 2)]
    for a, b in pairs:
        concept = find_common_concept(a["pattern"], b["pattern"])
        if not concept:
            continue
        val_a = pick_concept_across_years(a["pattern"], concept)
        val_b = pick_concept_across_years(b["pattern"], concept)
        if not val_a or not val_b:
            continue  # DISCARD -- one side has no real data, unanswerable premise
        q = generate_comparative_question(a["name"], b["name"], concept)
        rows.append({
            "id": _next_id(f"{a['ticker']}_{b['ticker']}_cmp"), "company": f"{a['ticker']},{b['ticker']}",
            "type": "comparative_cross_company", "question": q, "answer": None,
            "reference_context": {"a": val_a, "b": val_b},
            "verified": False, "note": "known-unbuilt: requires cross-company tool scoping",
        })
    return rows


def build_manual_stub_categories():
    stubs = []
    for c in COMPANIES[:6]:
        stubs.append({
            "id": _next_id(f"{c['ticker']}_analytical"), "company": c["ticker"],
            "type": "analytical_synthesis", "question": None, "answer": None,
            "reference_facts": [], "reference_narrative": [], "verified": False,
            "note": "author by hand -- see analytical_synthesis discussion",
        })
    for c in COMPANIES[:4]:
        stubs.append({
            "id": _next_id(f"{c['ticker']}_contradiction"), "company": c["ticker"],
            "type": "contradiction_reconciliation", "question": None, "answer": None,
            "reference_context": [], "verified": False,
            "note": "mine from real precision/segment mismatches",
        })
    for c in COMPANIES[:4]:
        stubs.append({
            "id": _next_id(f"{c['ticker']}_refusal"), "company": c["ticker"],
            "type": "unanswerable_refusal", "question": None, "answer": "not disclosed / not answerable",
            "reference_context": None, "verified": False,
            "note": "half 'not disclosed', half 'shouldn't editorialize'",
        })
    return stubs


if __name__ == "__main__":
    all_rows = (
        build_sql_only() + build_vector_only() + build_trend_multi_year()
        + build_comparative() + build_manual_stub_categories()
    )
    with open("eval/golden_set_draft.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"{len(all_rows)} rows written")

    # real validation pass, not just a count
    missing_company = [r["id"] for r in all_rows if r.get("question") and r["company"] and
                        not any(c["name"].lower() in r["question"].lower() for c in COMPANIES if c["ticker"] in r["company"])]
    print(f"rows still missing company name: {len(missing_company)} -> {missing_company}")