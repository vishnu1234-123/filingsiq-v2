from sentence_transformers import CrossEncoder
from eval.retrieval_hybrid import hybrid_search

_reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def reranked_search(query:str,company:str,company_filename:str,k:int=8,rerank_top:int=20)->list[str]:
    candidates=hybrid_search(query,company,company_filename,k=rerank_top)
    if not candidates:
        return []
    pairs=[[query,c] for c in candidates]
    scores=_reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:k]]