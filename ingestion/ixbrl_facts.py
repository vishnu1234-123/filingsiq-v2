from lxml import etree
import os


def _build_context_lookup(tree) -> dict:
    lookup = {}
    for ctx in tree.xpath("//*[name()='xbrli:context']"):
        ctx_id = ctx.get("id")
        start = ctx.xpath(".//*[name()='xbrli:startdate']")
        end = ctx.xpath(".//*[name()='xbrli:enddate']")
        instant = ctx.xpath(".//*[name()='xbrli:instant']")
        if start and end:
            lookup[ctx_id] = {"type": "duration", "start": start[0].text, "end": end[0].text}
        elif instant:
            lookup[ctx_id] = {"type": "instant", "date": instant[0].text}
        else:
            lookup[ctx_id] = {"type": "unknown"}
    return lookup


def _build_unit_lookup(tree) -> dict:
    lookup = {}
    for unit in tree.xpath("//*[name()='xbrli:unit']"):
        unit_id = unit.get("id")
        measure = unit.xpath(".//*[name()='xbrli:measure']")
        lookup[unit_id] = measure[0].text if measure else None
       
    return lookup


def _scaled_value(fact) -> float:
    raw = "".join(fact.itertext()).replace(",", "").strip()
    try:
        value = float(raw) if raw else 0.0
    except ValueError:
        return None
    scale = fact.get("scale")
    if scale and scale.isdigit():
        value *= 10 ** int(scale)
    if fact.get("sign") == "-":
        value = -value
    return value


def extract_numeric_facts(filepath: str) -> list[dict]:
    tree = etree.parse(filepath, etree.HTMLParser())
    context_lookup = _build_context_lookup(tree)
    unit_lookup = _build_unit_lookup(tree)

    seen = set()
    facts = []
    for tag in tree.xpath("//*[name()='ix:nonfraction']"):
        concept = tag.get("name")
        period = context_lookup.get(tag.get("contextref"), {})
        
        unit = unit_lookup.get(tag.get("unitref"))
        if unit is None:
            print("missing unit for concept:", concept)
        value = _scaled_value(tag)
        if value is None:
            continue
        key = (concept, value, period.get("start"), period.get("end"), period.get("date"))
        if key in seen:
            continue
        seen.add(key)
        facts.append({
            "concept": concept,
            "value": value,
            "unit": unit,
            "period_start": period.get("start"),
            "period_end": period.get("end"),
            "instant_date": period.get("date"),
        })
    return facts


if __name__ == "__main__":
    filepath = "../data/raw/aapl-20250927.htm"
    if not os.path.exists(filepath):
        print("File not found:", filepath)
        exit(1)
    numeric_facts = extract_numeric_facts(filepath)
    rnd = [f for f in numeric_facts if f["concept"] and "ResearchAndDevelopment" in f["concept"]]
    for f in rnd:
        print(f)
        