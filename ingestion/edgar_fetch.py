import requests

HEADERS={"User-Agent": "filingiq-v2 vishnuvardhan1920@gmail.com"}

def resolve_cik(ticker:str)->str|None:
    resp=requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS)
    resp.raise_for_status()
    for row in resp.json().values():
        if row["ticker"].upper()==ticker.upper():
            return str(row["cik_str"]).zfill(10)
    return None

def latest_filing_url(cik:str,form_type:str="10-K")->dict|None:
    resp = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=HEADERS)
    resp.raise_for_status()
    recent=resp.json()["filings"]["recent"]

    for i,form in enumerate(recent["form"]):
        if form == form_type:
            accession=recent["accessionNumber"][i].replace("-","")
            doc=recent["primaryDocument"][i]
            return {
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}",
                "accession_number": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "form_type": form,
            }
    return None

if __name__ == "__main__":
    ticker = "TSLA"
    cik = resolve_cik(ticker)
    print("resolved CIK:", cik)
    if cik:
        info = latest_filing_url(cik)
        print(info)