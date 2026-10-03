import re
from lxml import html,etree

HEADING_PATTERN = re.compile(r'^Item\s+\d+[A-Za-z]?\.', re.IGNORECASE)

def normalize_title(text:str)->str:
    text=text.replace("\u2019", "'").replace("\u2018", "'")
    text=re.sub(r"\s+"," ",text)
    return text.strip().lower().rstrip(".")

_RAW_TITLES = {
    "Business", "Risk Factors", "Unresolved Staff Comments", "Cybersecurity",
    "Properties", "Legal Proceedings", "Mine Safety Disclosures",
    "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "Quantitative and Qualitative Disclosures About Market Risk",
    "Financial Statements and Supplementary Data",
    "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "Controls and Procedures", "Other Information",
    "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "Directors, Executive Officers and Corporate Governance", "Executive Compensation",
    "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "Certain Relationships and Related Transactions, and Director Independence",
    "Principal Accountant Fees and Services",
    "Exhibits and Financial Statement Schedules", "Form 10-K Summary",
}
KNOWN_TITLES = {normalize_title(t) for t in _RAW_TITLES}
NOTE_PATTERN = re.compile(r'^Note\s+\d+\s*[–-]', re.IGNORECASE)
def _is_heading(el)->bool:
    if el.tag!="span" or not el.text:
        return False
    style=el.get("style") or ""
    if "font-weight:700" not in style:
        return False
    raw=el.text.strip()
    if HEADING_PATTERN.match(raw) or NOTE_PATTERN.match(raw):
        return True
    parent=el.getparent()
    if normalize_title(raw) in KNOWN_TITLES and parent is not None and len(list(parent))==1:
        return True

    if normalize_title(raw) in KNOWN_TITLES:
        row=el.getparent()
        while row is not None and row.tag!="tr":
            row=row.getparent()
        if row is not None:
            row_text="".join(row.itertext())
            if re.search(r"item\s+\d+[a-z]?\.", row_text, re.IGNORECASE):
                return True
    return False



def _extract_table_text(table_el)->str:
    rows=[]
    for tr in table_el.xpath(".//*[name()='tr']"):
        cells = tr.xpath("./*[name()='td'] | ./*[name()='th']")
        texts = ["".join(c.itertext()).strip() for c in cells]
        texts = [t for t in texts if t]
        if texts:
            rows.append(" | ".join(texts))
    return "\n".join(rows)
def _table_contains_heading(table_el) -> bool:
    for span in table_el.xpath(".//*[name()='span']"):
        if _is_heading(span):
            return True
    return False

def _walk(el,buffer):
    if _is_heading(el):
        buffer.append(("HEADING",el.text.strip()))
        if el.tail:
            buffer.append(("TEXT",el.tail))
        return 
    if el.tag=="table":
        if _table_contains_heading(el):
            if el.text:
                buffer.append(("TEXT", el.text))
            for child in el:
                _walk(child, buffer)
            if el.tail:
                buffer.append(("TEXT", el.tail))
            return
        buffer.append(("TABLE", _extract_table_text(el)))
        if el.tail:
            buffer.append(("TEXT", el.tail))
        return
    if el.text:
        buffer.append(("TEXT",el.text))
    for child in el:
        _walk(child,buffer)
    if el.tail:
        buffer.append(("TEXT",el.tail))


def split_into_sections(tree)->list[dict]:
    buffer=[]
    _walk(tree.getroot(),buffer)

    sections=[]
    current_heading=None
    current_blocks=[]

    for kind,content in buffer:
        if kind=="HEADING":
            if current_heading is not None or current_blocks:
                sections.append({"heading":current_heading,"blocks":current_blocks})
            current_heading=content
            current_blocks=[]
        else:
            current_blocks.append((kind,content))
    if current_heading is not None or current_blocks:
        sections.append({"heading":current_heading,"blocks":current_blocks})
    return sections




    
if __name__ == "__main__":
    import os
    filepath = "../data/raw/aapl-20250927.htm"
    if not os.path.exists(filepath):
        print("File not found:", filepath)
        exit(1)
    tree = etree.parse(filepath, etree.HTMLParser())
    sections = split_into_sections(tree)
    print(f"total sections: {len(sections)}")

    for s in sections:
        if s["heading"] and "financial statements" in s["heading"].lower():
            for kind, content in s["blocks"]:
                if kind == "TABLE" and "5,820" in content:
                    print(content)
                    break
            break
    
    print(sections[1]["heading"])


