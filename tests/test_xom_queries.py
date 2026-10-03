import sqlite3
from embed_and_store import get_collection,_MODEL

TEST_CASES = [
    {
        "question": "What was ExxonMobil's effective tax rate for 2025?",
        "sql_keyword": "EffectiveIncomeTaxRate",
    },
    {
        "question": "How much did ExxonMobil pay in dividends to shareholders in 2025?",
        "sql_keyword": "PaymentsOfDividends",
    },
    {
        "question": "How many shares did ExxonMobil issue to acquire Pioneer Natural Resources?",
        "sql_keyword": "StockIssued",
    },
    {
        "question": "What does ExxonMobil say about risks from climate change and the energy transition?",
        "sql_keyword": None,  # prose question, no single fact expected
    },
    {
        "question": "How does ExxonMobil manage cybersecurity risk?",
        "sql_keyword": None,
    },
    {
        "question": "What legal proceedings is ExxonMobil currently involved in?",
        "sql_keyword": None,
    },
    {
        "question": "How much did ExxonMobil spend on share repurchases in 2025, and what did they say was the reasoning?",
        "sql_keyword": "PaymentsForRepurchase",
    },
    {
        "question": "What happened with ExxonMobil's Pioneer acquisition?",
        "sql_keyword": "BusinessCombination",
    },
]

def check_sql(keyword:str):
    if not keyword:
        return []
    conn=sqlite3.connect("../data/facts.sqlite")
    cur=conn.execute(
        "SELECT concept,value,period_end,instant_date,segments_json "
        "FROM facts WHERE source_file LIKE '%xom%' AND concept LIKE ? LIMIT 5",
        (f"%{keyword}%",),
    )

    rows=cur.fetchall()
    conn.close()
    return rows

def check_vector(question:str,collection,n_results:int=3):
    query_embedding=_MODEL.encode([question]).tolist()
    results=collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where={"source_file": {"$eq": "../data/raw/xom-20251231.htm"}},
    )
    return list(zip(results["documents"][0], results["metadatas"][0]))

if __name__=="__main__":
    collection=get_collection(name="filing_chunks_mpnet")
    
    for case in TEST_CASES:
        print("\n" + "=" * 80)
        print("Q:", case["question"])

        sql_rows = check_sql(case["sql_keyword"])
        if case["sql_keyword"]:
            print(f"\n-- SQL check (concept LIKE '%{case['sql_keyword']}%') --")
            if sql_rows:
                for r in sql_rows:
                    print(r)
            else:
                print("  (no matching facts found)")

        print("\n-- Vector search top 3 --")
        for doc, meta in check_vector(case["question"], collection):
            print(f"[{meta['heading']}] ({meta['type']})")
            print(" ", doc[:150].replace("\n", " "))
    """
    from embed_and_store import get_collection
    c = get_collection(name="filing_chunks_mpnet")
    all_items = collection.get(include=["metadatas"])
    sample = [m["source_file"] for m in all_items["metadatas"] if "xom" in m.get("source_file", "")][:1]
    print(sample)
    """