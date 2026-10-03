import json
from collections import Counter

rows = json.load(open("eval/golden_set_draft.json"))
counts = Counter(r["type"] for r in rows)
for t, n in counts.items():
    print(t, n)

import json
rows = json.load(open("eval/golden_set_draft.json"))
vector_companies = [r["company"] for r in rows if r["type"] == "vector_only"]
from collections import Counter
print(Counter(vector_companies))



