import chromadb
from sentence_transformers import SentenceTransformer
import os

_THIS_DIR=os.path.dirname(os.path.abspath(__file__))

_MODEL=SentenceTransformer("all-mpnet-base-v2")

def get_collection(db_path:str=None,name:str="filing_chunks"):
    if db_path is None:
        db_path=os.path.join(_THIS_DIR,"..","data","chroma_store")
    client=chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=name)

def embed_and_store(collection,source_file:str,cik:str,chunks:list[dict])->None:
    texts=[c["text"] for c in chunks]
    if not texts:
        return 

    embeddings=_MODEL.encode(texts,show_progress_bar=False).tolist()
    ids=[ f"{source_file}::{i}" for i in range(len(chunks))]

    metadatas=[{
        "cik":cik or "",
        "source_file":source_file,
        "heading":c["heading"] or "",
        "type":c["type"],
        "token_count":c["token_count"],
        "chunk_index":i,
    } for i,c in enumerate(chunks)]

    collection.upsert(ids=ids,embeddings=embeddings,documents=texts,metadatas=metadatas)

if __name__=="__main__":
    import os
    from lxml import etree
    from section_splitter import split_into_sections
    from chunker import chunk_sections
    from ixbrl_facts import extract_numeric_facts

    filepath = "../data/raw/xom-20251231.htm"
    tree = etree.parse(filepath, etree.HTMLParser())
    sections = split_into_sections(tree)
    chunks = chunk_sections(sections)
    numeric = extract_numeric_facts(filepath)
    cik = numeric["facts"][0]["cik"] if numeric["facts"] else None

    collection = get_collection(name="filing_chunks_mpnet")  # same collection as Apple, multi-company
    embed_and_store(collection, filepath, cik, chunks)
    print("sections:", len(sections), "chunks:", len(chunks), "numeric facts:", len(numeric["facts"]), "skipped:", len(numeric["skipped"]))
    all_items = collection.get(include=["metadatas"], limit=1, where={"source_file": {"$contains": "xom"}}) if hasattr(collection, "get") else None
# safer, unfiltered check:
    sample = [m["source_file"] for m in all_items["metadatas"] if "xom" in m.get("source_file", "")][:1]
    print(sample)
