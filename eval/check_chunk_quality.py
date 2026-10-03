import json,re
from ingestion.embed_and_store import get_collection

def check_span_intactness(golden_path="eval/golden_set_draft.json"):
    rows=json.load(open(golden_path))
    collection=get_collection(name="filing_chunks_mpnet")
    all_chunks=collection.get(include=["documents","metadatas"])

    results=[]
    for row in rows:
        ref_candidates=[]
        if isinstance(row.get("reference_context"),str):
            ref_candidates.append(row["reference_context"])
        if isinstance(row.get("reference_narrative"),list):
            ref_candidates.extend(row["reference_narrative"])
        for ref in ref_candidates:
            if re.match(r"^[a-z]+:\w+ = ", ref) or re.match(r"^us-gaap:\w+ = ", ref):
                continue
            company_chunks=[d for d,m in zip(all_chunks["documents"],all_chunks["metadatas"])
                        if row["company"].split(",")[0] in m.get("source_file","")]
            intact=any(ref[:150].lower() in c.lower() for c in company_chunks)
            results.append({"id": row["id"], "intact": intact,"ref_preview":ref[:80]})
    intact_count=sum(r["intact"] for r in results)
    print(f"{intact_count}/{len(results)} vector_only rows have fully intact reference chunks")
    for r in results:
        if not r["intact"]:
            print("FRAGMENTED:", r["id"])
    return results

if __name__ == "__main__":
    check_span_intactness()