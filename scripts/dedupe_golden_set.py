"""
scripts/dedupe_golden_set.py
 
One-time cleanup for a specific, now-understood corruption: a broken
entry point in prepare_golden_set.py silently re-merged
golden_set_additions.json into golden_set_draft.json on every run
instead of running the real split logic, duplicating the same rows each
time. This removes duplicate ids, keeping the first occurrence of each,
and reports exactly what it removed so nothing silently disappears.
 
Usage:
    python -m scripts.dedupe_golden_set
    python -m scripts.dedupe_golden_set --dry-run   # report only, don't write
"""
 
import argparse
import json
from collections import Counter
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="eval/golden_set_draft.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
 
    path = Path(args.input)
    rows = json.loads(path.read_text())
    print(f"Loaded {len(rows)} rows from {args.input}")
 
    id_counts = Counter(r["id"] for r in rows)
    duplicated_ids = {id_: count for id_, count in id_counts.items() if count > 1}
 
    if not duplicated_ids:
        print("No duplicate ids found -- nothing to do.")
        return
 
    print(f"\nFound {len(duplicated_ids)} duplicated ids:")
    for id_, count in sorted(duplicated_ids.items()):
        print(f"  {id_!r}: appears {count} times")
 
    seen = set()
    deduped = []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        deduped.append(r)
 
    removed = len(rows) - len(deduped)
    print(f"\nRemoving {removed} duplicate rows -- {len(rows)} -> {len(deduped)}")
 
    if args.dry_run:
        print("\n--dry-run: not writing anything. Re-run without --dry-run to apply.")
        return
 
    backup_path = path.with_suffix(".json.bak")
    backup_path.write_text(json.dumps(rows, indent=2))
    print(f"Backup of the original (pre-dedup) file written to {backup_path}")
 
    path.write_text(json.dumps(deduped, indent=2))
    print(f"Wrote deduplicated file to {path}")
 
 
if __name__ == "__main__":
    main()
 
