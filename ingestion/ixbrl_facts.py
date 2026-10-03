from lxml import etree
import os
from  word2number import w2n


# ------------------------------------------------------------
# CONTEXT LOOKUP
# ------------------------------------------------------------
def _build_context_lookup(tree) -> dict:
    lookup = {}
    for ctx in tree.xpath("//*[name()='xbrli:context']"):
        ctx_id = ctx.get("id")

        start = ctx.xpath(".//*[name()='xbrli:startdate']")
        end = ctx.xpath(".//*[name()='xbrli:enddate']")
        instant = ctx.xpath(".//*[name()='xbrli:instant']")

        # CIK
        cik_el = ctx.xpath(".//*[name()='xbrli:identifier']")
        cik = cik_el[0].text if cik_el else None

        # MULTI-MEMBER SEGMENTS
        members = ctx.xpath(".//*[name()='xbrldi:explicitmember']")
        segments = tuple(sorted(
            (m.get("dimension"), m.text) for m in members
        ))

        # PERIOD TYPE
        if start and end:
            entry = {"type": "duration", "start": start[0].text, "end": end[0].text}
        elif instant:
            entry = {"type": "instant", "date": instant[0].text}
        else:
            entry = {"type": "unknown"}

        entry["cik"] = cik
        entry["segments"] = segments
        lookup[ctx_id] = entry

    return lookup


# ------------------------------------------------------------
# UNIT LOOKUP
# ------------------------------------------------------------
def _build_unit_lookup(tree) -> dict:
    lookup = {}
    for unit in tree.xpath("//*[name()='xbrli:unit']"):
        unit_id = unit.get("id")
        measure = unit.xpath(".//*[name()='xbrli:measure']")
        lookup[unit_id] = measure[0].text if measure else None
    return lookup


# ------------------------------------------------------------
# NUMERIC SCALING
# ------------------------------------------------------------
_ZERO_WORDS = {"no", "none"}  

def _scaled_value(fact):
    raw = "".join(fact.itertext()).replace(",", "").strip()

    if raw == "—" or fact.get("format") == "ixt:fixed-zero":
        return 0.0, None

    if fact.get("format") == "ixt-sec:numwordsen":
        word = raw.lower()
        if word in _ZERO_WORDS:
            return 0.0,None
        try:
            return float(w2n.word_to_num(word)),None
        except ValueError:
            return None, f"unrecognized number word: {raw!r}"

    try:
        value = float(raw) if raw else 0.0
    except ValueError:
        return None, f"unconvertible raw text: {raw!r}"
    scale = fact.get("scale")
    if scale and scale.isdigit():
        value *= 10 ** int(scale)
    if fact.get("sign") == "-":
        value = -value
    return value, None


# ------------------------------------------------------------
# NUMERIC FACT EXTRACTION
# ------------------------------------------------------------
def extract_numeric_facts(filepath: str) -> dict:
    tree = etree.parse(filepath, etree.HTMLParser())
    context_lookup = _build_context_lookup(tree)
    unit_lookup = _build_unit_lookup(tree)

    seen = set()
    facts = []
    skipped = []

    for tag in tree.xpath("//*[name()='ix:nonfraction']"):
        concept = tag.get("name")
        period = context_lookup.get(tag.get("contextref"), {})
        unit = unit_lookup.get(tag.get("unitref"))
        raw_text = "".join(tag.itertext()).strip()

        value, error = _scaled_value(tag)

        if value is None:
            skipped.append({
                "id": tag.get("id"),
                "concept": concept,
                "raw_text": raw_text,
                "reason": error,
            })
            continue

        segments = period.get("segments", ())

        # Dedup key includes segments
        key = (
            concept,
            value,
            period.get("start"),
            period.get("end"),
            period.get("date"),
            segments
        )

        if key in seen:
            continue
        seen.add(key)

        facts.append({
            "concept": concept,
            "value": value,
            "raw_text": raw_text,
            "unit": unit,
            "decimals": tag.get("decimals"),
            "period_start": period.get("start"),
            "period_end": period.get("end"),
            "instant_date": period.get("date"),
            "segments": [{"dimension": d, "member": m} for d, m in segments],
            "cik": period.get("cik"),
            "source": "xbrl",
        })

    return {"facts": facts, "skipped": skipped}


# ------------------------------------------------------------
# TEXT FACT EXTRACTION
# ------------------------------------------------------------
def extract_text_facts(filepath: str) -> list[dict]:
    tree = etree.parse(filepath, etree.HTMLParser())
    facts = []

    for tag in tree.xpath("//*[name()='ix:nonnumeric']"):
        concept = tag.get("name")

        # Skip TextBlocks (prose)
        if concept and concept.endswith("TextBlock"):
            continue

        facts.append({
            "concept": concept,
            "value": "".join(tag.itertext()).strip(),
            "source": "xbrl",
        })

    return facts


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    filepath = "../data/raw/xom-20251231.htm"

    if not os.path.exists(filepath):
        print("File not found:", filepath)
        exit(1)

    numeric = extract_numeric_facts(filepath)
    cik = numeric["facts"][0]["cik"] if numeric["facts"] else None
    text_facts = extract_text_facts(filepath)
    """
    print(f"numeric facts: {len(result['facts'])}")
    print(f"skipped facts: {len(result['skipped'])}")
    print(f"text facts (TextBlocks excluded): {len(text_facts)}")

    print("\nany skipped facts:")
    for s in result["skipped"][:10]:
        print(s)

    print("\nrevenue facts with segments field:")
    revenue = [
        f for f in result["facts"]
        if f["concept"] and "RevenueFromContractWithCustomer" in f["concept"]
    ]
    for f in revenue[:3]:
        print(f["value"], f["segments"])
    
    net_income = [f for f in result["facts"] if f["concept"] == "us-gaap:NetIncomeLoss"]
    total_assets = [f for f in result["facts"] if f["concept"] == "us-gaap:Assets"]

    for f in net_income:
        print(f["value"], f["period_end"], f["segments"])
    for f in total_assets:
        print(f["value"], f["instant_date"], f["segments"])

    print("\nconsolidated total revenue (should have empty segments):")
    total = [f for f in result["facts"] if f["value"] == 416161000000.0]
    for f in total:
        print(f["value"], f["segments"])
    """
    from db_loader import get_connection, init_db, insert_facts
    from embed_and_store import get_collection, embed_and_store
    from section_splitter import split_into_sections
    from chunker import chunk_sections

    conn = get_connection()
    init_db(conn)
    insert_facts(conn, filepath, numeric)

    tree = etree.parse(filepath, etree.HTMLParser())
    sections = split_into_sections(tree)
    chunks = chunk_sections(sections)

    collection = get_collection(name="filing_chunks_mpnet")
    embed_and_store(collection, filepath, cik, chunks)

    cur = conn.execute("SELECT COUNT(*) FROM facts WHERE source_file LIKE '%xom%'")
    print("facts inserted:", cur.fetchone())
    print("chunks embedded:", len(chunks))
    