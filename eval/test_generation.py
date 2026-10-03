from eval.retrieval_hybrid import hybrid_search
chunks = hybrid_search("Who is MetLife's auditor?", "met", "met-20251231.htm", k=5)
for c in chunks:
    if "REPORT OF INDEPENDENT" in c:
        print(c[:600])