# normalize_paths.py -- run once, from the project root
import sqlite3
import os

conn = sqlite3.connect("data/facts.sqlite")
rows = conn.execute("SELECT DISTINCT source_file FROM facts").fetchall()
for (old_path,) in rows:
    new_path = os.path.basename(old_path)
    if old_path != new_path:
        conn.execute("UPDATE facts SET source_file = ? WHERE source_file = ?", (new_path, old_path))
        print(f"renamed: {old_path} -> {new_path}")
conn.commit()
print("\nfinal state:")
print(conn.execute("SELECT DISTINCT source_file FROM facts").fetchall())