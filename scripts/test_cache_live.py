import argparse
import time 

from orchestration.router import answer_question

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--question",default="What was Apple's R&D spend in FY2025?")
    args=ap.parse_args()

    print(f"Question: {args.question!r}\n")

    print("--- Call 1 (expect: full pipeline, real cost, real latency) ---")
    t0=time.monotonic()
    r1=answer_question(args.question)
    t1=time.monotonic()-t0
    print(f"answer: {r1['answer']}")
    print(f"faithfulness: {r1['faithfulness']}  retries: {r1['retry_count']}")
    print(f"cost: ${r1['cost_summary']['total_cost_usd']}  latency: {t1:.2f}s\n")

    print("--- Call 2, same question (expect: cache hit, $0 cost, near-instant) ---")
    t0 = time.monotonic()
    r2 = answer_question(args.question)
    t2 = time.monotonic() - t0
    print(f"answer: {r2['answer']}")
    print(f"cost: ${r2['cost_summary']['total_cost_usd']}  latency: {t2:.2f}s\n")

    print("="*60)
    if r2["cost_summary"]["total_cost_usd"]==0 and t2<t1/2:
        print(f"CACHE WORKING: call 2 cost $0 and was {t1/max(t2,0.001):.1f}x faster.")
    else:
        print("CACHE NOT HIT -- call 2 still cost money / took similar time.")
        print("Check: is redis-server actually running? Does REDIS_URL match where it's listening?")
    
    print(f"\nNow run: python -m scripts.view_trace --last 2")
    print("Call 1's trace should show the full pipeline. Call 2's should show 'SERVED FROM CACHE'.")
 
 
if __name__ == "__main__":
    main()