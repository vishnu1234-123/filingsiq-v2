from eval.retrieval_dense import dense_search
from eval.retrieval_sparse import sparse_search,_load_company_chunks
from rank_bm25 import BM25Okapi

def hybrid_search(sparse_query:str,company:str,company_filename:str,k:int=8,rrf_k:int=60,dense_query:str=None)->list[str]:
    dense_query=dense_query or sparse_query
    docs=_load_company_chunks(company_filename)
    if not docs:
        return []
    bm25 = BM25Okapi([d.lower().split() for d in docs])
    sparse_scores = bm25.get_scores(sparse_query.lower().split())
    sparse_ranked = sorted(range(len(docs)), key=lambda i: sparse_scores[i], reverse=True)[:k]

    dense_docs = dense_search(dense_query, company_filename, k)
    dense_ranked_idx = [docs.index(d) for d in dense_docs if d in docs]

    rrf_scores = {}
    for rank, idx in enumerate(sparse_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank)
    for rank, idx in enumerate(dense_ranked_idx):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [docs[i] for i, _ in merged]