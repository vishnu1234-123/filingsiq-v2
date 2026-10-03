# eval/retrieval_dense.py
from sentence_transformers import SentenceTransformer
from ingestion.embed_and_store import get_collection

_MODEL = SentenceTransformer("all-mpnet-base-v2")

def dense_search(query: str, company_filename: str, k: int = 8) -> list[str]:
    collection = get_collection(name="filing_chunks_mpnet")
    embedding = _MODEL.encode([query]).tolist()
    results = collection.query(
        query_embeddings=embedding, n_results=k,
        where={"source_file": {"$eq": company_filename}},
    )
    return results["documents"][0] if results["documents"] else []