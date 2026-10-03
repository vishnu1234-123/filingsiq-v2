from ingestion.embed_and_store import get_collection

def search_chunks(source_pattern:str,keywords:list[str]):
    collection=get_collection(name="filing_chunks_mpnet")
    items=collection.get(include=["documents","metadatas"])
    for doc,meta in zip(items["documents"],items["metadatas"]):
        if source_pattern in meta.get("source_file",""):
            if any(k.lower() in doc.lower() for k in keywords):
                print(f"[{meta.get('heading') or 'no heading'}]")
                print(doc[:600])
                print("---")

if __name__ == "__main__":
    search_chunks("xom", ["climate change", "net-zero"])
    search_chunks("wmt", ["customer or member perceptions"])