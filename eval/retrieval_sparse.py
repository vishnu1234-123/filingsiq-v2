from rank_bm25 import BM25Okapi
from ingestion.embed_and_store import get_collection

def _load_company_chunks(company_filename:str):
    collection=get_collection(name="filing_chunks_mpnet")
    items=collection.get(include=["documents","metadatas"])
    return [d for d,m in zip(items["documents"],items["metadatas"])
            if m.get("source_file")==company_filename]

def sparse_search(query:str,company_filename:str,k:int=8)->list[str]:
    docs=_load_company_chunks(company_filename)
    if not docs:
        return []
    bm25=BM25Okapi([d.lower().split() for d in docs])
    scores=bm25.get_scores(query.lower().split())
    ranked=sorted(range(len(docs)),key=lambda i:scores[i],reverse=True)[:k]
    return [docs[i] for i in ranked]
