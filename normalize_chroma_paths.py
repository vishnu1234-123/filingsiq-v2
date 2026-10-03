import chromadb
c = chromadb.PersistentClient(path="/Users/vishnuvardhan/Desktop/RAG AGENT/data/chroma_store")
collection = c.get_or_create_collection(name="filing_chunks_mpnet")
items = collection.get(include=["metadatas"])
from collections import Counter
print("total:", collection.count())
print(Counter(m.get("source_file", "MISSING") for m in items["metadatas"]))