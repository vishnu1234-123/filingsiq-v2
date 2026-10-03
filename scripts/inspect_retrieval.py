"""
scripts/inspect_retrieval.py

The audit log truncates result_preview to 200 chars everywhere -- fine
for a short SQL tuple, useless for auditing a 4000+ character vector
chunk. This prints the FULL retrieved text for a given company/question,
so you can just read it and see whether the fact you're looking for is
actually in there, instead of guessing from a 200-char snippet.

Usage:
    python -m scripts.inspect_retrieval aapl "What was Apple's R&D spend in FY2025?"
    python -m scripts.inspect_retrieval aapl "..." --k 10 --hyde
"""

import argparse

from orchestration.router import run_vector, run_sql


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("company")
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--hyde", action="store_true")
    ap.add_argument("--also-sql", action="store_true", help="also try the SQL path, to compare")
    args = ap.parse_args()

    print("=" * 70)
    print(f"VECTOR retrieval for: {args.question!r}  (company={args.company}, k={args.k}, hyde={args.hyde})")
    print("=" * 70)
    text, status = run_vector(args.company, args.question, k=args.k, use_hyde=args.hyde)
    print(f"status: {status}")
    print(f"length: {len(text)} chars\n")
    print(text)
    print()

    if args.also_sql:
        print("=" * 70)
        print("SQL path, for comparison")
        print("=" * 70)
        result, status = run_sql(args.company, args.question)
        print(f"status: {status}")
        print(result)


if __name__ == "__main__":
    main()