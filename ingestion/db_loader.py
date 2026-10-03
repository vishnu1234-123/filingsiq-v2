import sqlite3,os
import json
import hashlib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
def compute_file_hash(filepath:str)->str:
    """SHA-256 of the raw file bytes. Used to detect whether an amendment
    (10-K/A) or a re-download of the same filing is genuinely new content
    or byte-identical to something already ingested."""
    with open(filepath,"rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_connection(db_path:str="../data/facts.sqlite")->sqlite3.Connection:
    if db_path is None:
        db_path = os.path.join(_THIS_DIR, "..", "data", "facts.sqlite")
    return sqlite3.connect(db_path)

def init_db(conn: sqlite3.Connection)->None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cik TEXT,
            source_file TEXT,
            concept TEXT,
            value REAL,
            raw_text TEXT,
            unit TEXT,
            decimals TEXT,
            period_start TEXT,
            period_end TEXT,
            instant_date TEXT,
            segments_json TEXT,
            source TEXT,
            UNIQUE(cik,concept,value,period_start,period_end,instant_date,segments_json)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS skipped_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            tag_id TEXT,
            concept TEXT,
            raw_text TEXT,
            reason TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingested_files (
            source_file TEXT PRIMARY_KEY,
            file_hash TEXT,
            ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
            is_deprecated INTEGER DEFAULT 0
        )
    """)
    conn.commit()

def check_and_register_file(conn:sqlite3.Connection,filepath:str)->str:
    """
    Returns 'new', 'unchanged', or 'amended' -- tells the caller whether
    to proceed with ingestion, skip it entirely, or mark old facts
    deprecated before inserting the new version.
    """
    new_hash=compute_file_hash(filepath)
    cur=conn.execute(
        "SELECT file_hash FROM ingested_files WHERE source_file = ?",(filepath,)
    )
    row=cur.fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO ingested_files (source_file,file_hash) VALUES (?,?)",
            (filepath,new_hash),
        )
        conn.commit()
        return "new"
    elif row[0]==new_hash:
        return "unchanged"
    else:
        conn.execute(
            "UPDATE facts SET is_deprecated=1 WHERE source_file=? AND is_deprecated=0",
            (filepath,),
        )
        conn.execute(
            "UPDATE ingested_files SET file_hash = ?, is_deprecated = 0 WHERE source_file = ?",
            (new_hash,filepath),
        )
        conn.commit()
        return "ammended"

def insert_facts(conn:sqlite3.Connection,source_file:str,extraction_result:dict)->None:
    for f in extraction_result["facts"]:
        segments_json=json.dumps(f["segments"],sort_keys=True)
        conn.execute("""
            INSERT OR IGNORE INTO facts
            (cik, source_file, concept, value, raw_text, unit, decimals,
             period_start, period_end, instant_date, segments_json, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
            f["cik"] or "", source_file, f["concept"] or "", f["value"], f["raw_text"] or "",
            f["unit"] or "", f["decimals"] or "", f["period_start"] or "", f["period_end"] or "",
            f["instant_date"] or "", segments_json, f["source"] or "",
            )
        )
    for s in extraction_result["skipped"]:
        conn.execute("""
            INSERT INTO skipped_facts (source_file, tag_id, concept, raw_text, reason)
            VALUES (?, ?, ?, ?, ?)
        """, (source_file, s["id"], s["concept"], s["raw_text"], s["reason"]))

    conn.commit()

