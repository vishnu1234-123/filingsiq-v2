# ingestion/ingest_filing.py
import os
from lxml import etree

from ingestion.ixbrl_facts import extract_numeric_facts
from ingestion.section_splitter import split_into_sections
from ingestion.chunker import chunk_sections
from ingestion.db_loader import get_connection, init_db, insert_facts, check_and_register_file
from ingestion.embed_and_store import get_collection, embed_and_store


def canonical_source_id(filepath: str) -> str:
    return os.path.basename(filepath)


def _chunks_already_embedded(collection, source_id: str) -> bool:
    """Checks the vector store directly, independent of the facts-side
    idempotency check -- these two can genuinely be out of sync, as
    JPMorgan just proved."""
    existing = collection.get(where={"source_file": {"$eq": source_id}}, include=[])
    return len(existing["ids"]) > 0


def ingest_filing(filepath: str, collection_name: str = "filing_chunks_mpnet") -> dict:
    if not os.path.exists(filepath):
        return {"status": "error", "reason": "file not found"}

    source_id = canonical_source_id(filepath)
    conn = get_connection()
    init_db(conn)

    facts_status = check_and_register_file(conn, filepath)
    collection = get_collection(name=collection_name)
    needs_embedding = not _chunks_already_embedded(collection, source_id)

    result = {"facts_status": facts_status, "embedded_now": needs_embedding}

    if facts_status in ("new", "amended"):
        numeric = extract_numeric_facts(filepath)
        insert_facts(conn, source_id, numeric)
        result["facts_count"] = len(numeric["facts"])
        result["skipped_count"] = len(numeric["skipped"])
        cik = numeric["facts"][0]["cik"] if numeric["facts"] else None
    else:
        cik = None  # facts already correct, cik not needed unless we're embedding too

    if needs_embedding:
        tree = etree.parse(filepath, etree.HTMLParser())
        sections = split_into_sections(tree)
        chunks = chunk_sections(sections)
        embed_and_store(collection, source_id, cik, chunks)
        result["chunks_count"] = len(chunks)

    if facts_status == "unchanged" and not needs_embedding:
        result["status"] = "fully_up_to_date"
    else:
        result["status"] = "processed"

    return result


if __name__ == "__main__":
    import sys
    print(ingest_filing(sys.argv[1]))