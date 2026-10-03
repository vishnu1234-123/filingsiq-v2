import re
from typing import Optional

_MONTH_NAMES = "January|February|March|April|May|June|July|August|September|October|November|December"


def _extract_period(question: str) -> Optional[str]:
    match = re.search(rf"({_MONTH_NAMES})\s+(\d{{1,2}}),?\s+(\d{{4}})", question)
    if not match:
        return None
    from datetime import datetime
    try:
        dt = datetime.strptime(f"{match.group(1)} {match.group(2)} {match.group(3)}", "%B %d %Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _extract_fiscal_year(question: str) -> Optional[str]:
    match = re.search(
        r"\bFY\s?(\d{4})\b|\bfiscal (?:year )?(\d{4})\b|\b(?:in|for|during)\s+(\d{4})\b",
        question, re.IGNORECASE
    )
    if not match:
        return None
    return match.group(1) or match.group(2) or match.group(3)


def run_sql_period_logic(sub_q: str, facts: list, picked: str) -> tuple[str, str]:
    """Isolated copy of just the period-filtering section of run_sql, with
    candidate_concepts/_pick_concept/resolve_concept already resolved
    (facts, picked passed in directly) -- tests only the logic in question."""
    period = _extract_period(sub_q)
    if period:
        period_matched = [f for f in facts if f[2] == period or f[3] == period]
        if period_matched:
            facts = period_matched
        else:
            return (
                f"No data for {picked!r} on {period} -- not covered "
                f"(available: {sorted(set(f[2] or f[3] for f in facts))}).",
                "permanent_failure",
            )
    else:
        fiscal_year = _extract_fiscal_year(sub_q)
        if fiscal_year:
            year_matched = [f for f in facts if f[2].startswith(fiscal_year) or f[3].startswith(fiscal_year)]
            if year_matched:
                facts = year_matched
            else:
                return (
                    f"No data for {picked!r} in fiscal year {fiscal_year} -- not covered "
                    f"(available years: {sorted(set((f[2] or f[3])[:4] for f in facts))}).",
                    "permanent_failure",
                )
    unsegmented = [f for f in facts if f[4] == "[]"]
    segmented = [f for f in facts if f[4] != "[]"]
    ordered = unsegmented + segmented
    return "\n".join(str(f) for f in ordered[:8]), "ok"


# Realistic fact rows, matching the real trace's actual shape
AAPL_REVENUE_FACTS = [
    ('us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax', 416161000000.0, '2025-09-27', '', '[]'),
    ('us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax', 391035000000.0, '2024-09-28', '', '[]'),
    ('us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax', 383285000000.0, '2023-09-30', '', '[]'),
]

passed, failed = 0, 0
def check(name, cond):
    global passed, failed
    if cond: passed += 1; print(f"  PASS  {name}")
    else: failed += 1; print(f"  FAIL  {name}")

# THE ACTUAL BUG CASE: bare year, not in data -- must now be permanent_failure
result, status = run_sql_period_logic("what is apple revenue in 2022", AAPL_REVENUE_FACTS, "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax")
check("2022 (not in data, bare year) -> permanent_failure", status == "permanent_failure")
check("2022 failure message names the missing year", "2022" in result)

# REGRESSION: bare year that DOES exist should still work correctly
result, status = run_sql_period_logic("what is apple revenue in 2025", AAPL_REVENUE_FACTS, "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax")
check("2025 (bare year, IS in data) -> ok", status == "ok")
check("2025 result correctly filtered to just that row", "416161000000.0" in result and "391035000000.0" not in result)

# REGRESSION: existing FY-prefix phrasing still works
result, status = run_sql_period_logic("what is apple revenue in FY2024", AAPL_REVENUE_FACTS, "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax")
check("FY2024 (existing phrasing, IS in data) -> ok", status == "ok")
check("FY2024 result correctly filtered", "391035000000.0" in result and "416161000000.0" not in result)

# REGRESSION: FY-prefix for a year NOT in data
result, status = run_sql_period_logic("what is apple revenue in FY2020", AAPL_REVENUE_FACTS, "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax")
check("FY2020 (existing phrasing, NOT in data) -> permanent_failure", status == "permanent_failure")

# REGRESSION: no year mentioned at all -- unfiltered behavior preserved
result, status = run_sql_period_logic("what is apple revenue", AAPL_REVENUE_FACTS, "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax")
check("no year mentioned -> ok, unfiltered (all 3 rows present)", status == "ok" and all(str(f[1]) in result for f in AAPL_REVENUE_FACTS))

# Should NOT falsely match "from 2023 to 2024" as a single bare year
check("'from 2023 to 2024' does not match the bare-year pattern", _extract_fiscal_year("change from 2023 to 2024") is None)

print(f"\n{passed} passed, {failed} failed")