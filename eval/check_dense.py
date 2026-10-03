import json
rows = json.load(open("eval/golden_set_draft.json"))

vector_only_count = len([r for r in rows if r["type"] == "vector_only"])
analytical_count = len([r for r in rows if r["type"] == "analytical_synthesis"])
print("vector_only:", vector_only_count)
print("analytical_synthesis:", analytical_count)
print("sum:", vector_only_count + analytical_count)

import json
rows = json.load(open("eval/golden_set_draft.json"))

# replicate check_chunk_quality's actual counting logic exactly,
# to see which specific rows it counted that this simple sum doesn't
for row in rows:
    if row["type"] in ("vector_only", "analytical_synthesis"):
        continue
    ref = row.get("reference_context")
    narr = row.get("reference_narrative")
    if isinstance(ref, str) and ref.strip():
        print("unexpected string reference_context on non-vector/analytical row:", row["id"], row["type"])
    if isinstance(narr, list) and narr:
        print("unexpected reference_narrative on non-vector/analytical row:", row["id"], row["type"])