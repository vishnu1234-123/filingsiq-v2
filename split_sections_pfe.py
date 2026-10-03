from lxml import etree

tree = etree.parse("data/raw/pfe-20251231.htm", etree.HTMLParser())
all_spans = tree.getroot().xpath("//*[name()='span']")
print("total span tags found:", len(all_spans))

bold_spans = [s for s in all_spans if "font-weight:700" in (s.get("style") or "")]
print("bold spans found:", len(bold_spans))
if bold_spans:
    print("sample bold span text:", repr(bold_spans[0].text))

from ingestion.section_splitter import HEADING_PATTERN, NOTE_PATTERN, normalize_title, KNOWN_TITLES

for s in bold_spans:
    raw = (s.text or "").strip()
    if HEADING_PATTERN.match(raw):
        print("MATCHES HEADING_PATTERN:", repr(raw))
    elif normalize_title(raw) in KNOWN_TITLES:
        print("MATCHES KNOWN_TITLES:", repr(raw))

matches = [s for s in bold_spans if (s.text or "").strip() == "ITEM\xa01."]
span = matches[0]
parent = span.getparent()
print("parent tag:", parent.tag)
print("number of parent's children:", len(list(parent)))
print("parent's children tags:", [c.tag for c in parent])

# walk up to find enclosing <tr>, same logic as _is_heading
row = span.getparent()
while row is not None and row.tag != "tr":
    row = row.getparent()
print("found enclosing tr:", row is not None)