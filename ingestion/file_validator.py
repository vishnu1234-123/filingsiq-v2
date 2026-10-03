
def validate_filing_upload(filepath: str) -> tuple[bool, str]:
    if not filepath.lower().endswith((".htm", ".html", ".txt")):
        return False, "Unsupported file type. Please upload the .htm filing from EDGAR."
    with open(filepath, "rb") as f:
        head = f.read(200).lower()
    if b"%pdf" in head:
        return False, "This looks like a PDF renamed to .htm. Please upload the original .htm from EDGAR."
    return True, ""


if __name__ == "__main__":
    # test against real files you already have
    for path in ["../data/raw/aapl-20250927.htm", "../data/raw/jpm-20251231.htm"]:
        print(path, validate_filing_upload(path))