# check_collection.py
from ingestion.embed_and_store import get_collection
from collections import Counter

c = get_collection(name="filing_chunks_mpnet")
items = c.get(include=["metadatas"])
print("total items:", c.count())
print(Counter(m.get("source_file", "MISSING") for m in items["metadatas"]))