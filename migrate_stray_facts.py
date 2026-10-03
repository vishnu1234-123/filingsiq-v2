import sqlite3

REAL_DB = "/Users/vishnuvardhan/Desktop/RAG AGENT/filingsiq_v2/data/facts.sqlite"
STRAY_DB = "/Users/vishnuvardhan/Desktop/RAG AGENT/data/facts.sqlite"

conn=sqlite3.connect(REAL_DB)
conn.execute(f"ATTACH DATABASE '{STRAY_DB}' AS stray")

conn.execute("""
    INSERT OR IGNORE INTO facts
    (cik,source_file,concept,value,raw_text,unit,decimals,
     period_start,period_end,instant_date,segments_json,source)
    SELECT cik, source_file, concept, value, raw_text, unit, decimals,
           period_start, period_end, instant_date, segments_json, source
    FROM stray.facts
""")

conn.commit()

result = conn.execute("SELECT source_file, COUNT(*) FROM facts GROUP BY source_file").fetchall()
for row in result:
    print(row)

conn.execute("DETACH DATABASE stray")
conn.close()