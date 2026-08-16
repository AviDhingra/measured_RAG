import time
from qdrant_setup import get_client
from self_rag import self_aware_retrieve
from golden_queries import RetrievalQuery, load_queries
from evaluate_tier1 import precision_at_k, recall_at_k, evaluate, aggregate_by_category
from critic import critique_retrieval
from qdrant_setup import search

K = 5
MAX_RETRIES = 2

from pathlib import Path

TIER2_REPORT_PATH = Path("../tier2_report.md")


def evaluate_tier2(queries: list[RetrievalQuery], k: int = K, max_retries: int = MAX_RETRIES):
    client = get_client()
    scored, unanswerable = [], []

    for q in queries:
        start = time.perf_counter()
        result = self_aware_retrieve(client, q.query_text, k=k, max_retries=max_retries)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if q.category == "unanswerable":
            unanswerable.append({
                "query": q.query_text,
                "retrieved": result.final_chunk_ids,
                "attempts": result.attempts,
                "confident": result.confident,
            })
            continue

        relevant = set(q.relevant_chunk_ids)
        scored.append({
            "query_id": q.query_id,
            "category": q.category,
            "precision": precision_at_k(result.final_chunk_ids, relevant, k),
            "recall": recall_at_k(result.final_chunk_ids, relevant, k),
            "attempts": result.attempts,
            "latency_ms": elapsed_ms,
        })

    return scored, unanswerable


def cost_summary(scored: list[dict]) -> dict:
    total = len(scored)
    needed_retry = sum(1 for r in scored if r["attempts"] > 1)
    return {
        "pct_needed_retry": needed_retry / total * 100,
        "avg_attempts": sum(r["attempts"] for r in scored) / total,
        "avg_latency_ms": sum(r["latency_ms"] for r in scored) / total,
    }


def write_tier2_report(tier1_by_cat: dict, tier2_by_cat: dict, cost: dict, k: int) -> None:
    lines = [
        "# Arc A — Tier 2 vs. Tier 1",
        "",
        "## Precision@k / recall@k, by category",
        "",
        "| Category | Tier 1 precision | Tier 2 precision | Tier 1 recall | Tier 2 recall |",
        "|---|---|---|---|---|",
    ]
    for cat in sorted(tier2_by_cat):
        t1 = tier1_by_cat.get(cat, {"mean_precision": float("nan"), "mean_recall": float("nan")})
        t2 = tier2_by_cat[cat]
        lines.append(
            f"| {cat} | {t1['mean_precision']:.2f} | {t2['mean_precision']:.2f} "
            f"| {t1['mean_recall']:.2f} | {t2['mean_recall']:.2f} |"
        )

    lines += [
        "",
        "## Cost of self-awareness",
        "",
        f"- {cost['pct_needed_retry']:.0f}% of queries needed at least one retry",
        f"- average attempts per query: {cost['avg_attempts']:.2f}",
        f"- average latency per query: {cost['avg_latency_ms']:.0f}ms",
    ]

    TIER2_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    queries = load_queries()  # from Lesson A.4

    tier1_scored, _ = evaluate(queries)  # from Lesson A.5's evaluate_tier1.py, run against the plain retriever
    tier1_by_cat = aggregate_by_category(tier1_scored)  # also from Lesson A.5

    tier2_scored, tier2_unanswerable = evaluate_tier2(queries)
    tier2_by_cat = aggregate_by_category(tier2_scored)
    cost = cost_summary(tier2_scored)

    q06 = next(q for q in queries if q.query_id == "q06")
    client = get_client()

    results = search(client, q06.query_text, k=5)
    texts = [p.payload["text"] for p in results.points]
    print("retrieved:", [p.payload["chunk_id"] for p in results.points])
    print("should have been:", q06.relevant_chunk_ids)

    critique = critique_retrieval(q06.query_text, texts)
    print("reasoning:", critique.reasoning)
    

    write_tier2_report(tier1_by_cat, tier2_by_cat, cost, k=K)
    print(f"Tier 2 report written -> {TIER2_REPORT_PATH}")

    #for row in tier2_scored:
        #if row["category"] == "ambiguous":
            #print(row["query_id"], "attempts:", row["attempts"], "precision:", row["precision"], "recall:", row["recall"])