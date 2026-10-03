def reciprocal_rank(retrieved_texts:list[str],reference_text:str)->float:
    ref_snippet=reference_text[:150].lower()
    for i,text in enumerate(retrieved_texts):
        if ref_snippet in text.lower():
            return 1.0/(i+1)
    return 0

def recall_at_k(retrieved_texts:list[str],reference_text:str,k:int)->int:
    ref_snippet=reference_text[:150].lower()
    return int(any(ref_snippet in t.lower() for t in retrieved_texts[:k]))

