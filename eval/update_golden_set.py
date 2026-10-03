import json
draft = json.load(open("eval/golden_set_draft.json"))
additions = json.load(open("eval/golden_set_additions.json"))
json.dump(draft + additions, open("eval/golden_set_draft.json", "w"), indent=2)