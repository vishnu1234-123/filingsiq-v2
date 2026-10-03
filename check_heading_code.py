from lxml import etree
from ingestion.section_splitter import split_into_sections

for filepath in ["data/raw/pfe-20251231.htm", "data/raw/xom-20251231.htm"]:
    tree = etree.parse(filepath, etree.HTMLParser())
    sections = split_into_sections(tree)
    headings = [s["heading"] for s in sections if s["heading"]]
    print(f"\n{filepath}")
    print(f"  total sections: {len(sections)}")
    print(f"  non-empty headings: {len(headings)}")
    for h in headings[:5]:
        print("   ", repr(h))

from lxml import etree
from ingestion.section_splitter import split_into_sections

for ticker, filename in [("aapl", "aapl-20250927.htm"), ("jpm", "jpm-20251231.htm"),
                          ("wmt", "wmt-20250131.htm"), ("pld", "pld-20251231.htm"),
                          ("cat", "cat-20251231.htm"), ("met", "met-20251231.htm"),
                          ("ms", "ms-20251231.htm"), ("tsla", "tsla-20251231.htm")]:
    tree = etree.parse(f"data/raw/{filename}", etree.HTMLParser())
    sections = split_into_sections(tree)
    headings = [s["heading"] for s in sections if s["heading"]]
    print(f"{ticker}: {len(sections)} sections, {len(headings)} headings")

from ingestion.embed_and_store import get_collection

collection = get_collection(name="filing_chunks_mpnet")
for source in ["aapl-20250927.htm", "jpm-20251231.htm", "wmt-20250131.htm",
               "cat-20251231.htm", "met-20251231.htm", "ms-20251231.htm", "tsla-20251231.htm"]:
    collection.delete(where={"source_file": {"$eq": source}})
    print("deleted:", source)