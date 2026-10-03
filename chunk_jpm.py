from ingestion.embed_and_store import get_collection

collection = get_collection(name="filing_chunks_mpnet")
items = collection.get(include=["metadatas"])

for ticker in ["ms", "met", "cat", "tsla"]:
    matches = [m for m in items["metadatas"] if ticker in m.get("source_file", "")]
    headings = {m.get("heading", "") for m in matches}
    non_empty = [h for h in headings if h.strip()]
    print(f"{ticker}: {len(matches)} chunks, {len(non_empty)} non-empty headings")

