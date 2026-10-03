from ingestion.embed_and_store import get_collection

collection = get_collection(name="filing_chunks_mpnet")

before = len([m for m in collection.get(include=["metadatas"])["metadatas"] if "jpm" in m.get("source_file", "")])
print("before delete:", before)

collection.delete(where={"source_file": {"$eq": "jpm-20251231.htm"}})

after = len([m for m in collection.get(include=["metadatas"])["metadatas"] if "jpm" in m.get("source_file", "")])
print("after delete:", after)