import time

from golden_queries import RetrievalQuery, load_queries
from qdrant_setup import get_client, search
from pathlib import Path

REPORT_PATH = Path("../tier1_report.md")

K = 5

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for c in top_k if c in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("recall@k is undefined for a query with no relevant chunks")
    top_k = retrieved[:k]
    hits = sum(1 for c in top_k if c in relevant)
    return hits / len(relevant)


#assert precision_at_k(["c1", "c3", "c5"], {"c1", "c2", "c5", "c7"}, 3) == 2 / 3
#assert recall_at_k(["c1", "c3", "c5"], {"c1", "c2", "c5", "c7"}, 3) == 2 / 4


def retrieve_chunk_ids(client, query_text: str, k: int) -> tuple[list[str], float]:
    start = time.perf_counter()
    results = search(client, query_text, k=k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return [p.payload["chunk_id"] for p in results.points], elapsed_ms





def evaluate(queries: list[RetrievalQuery], k: int = K):
    client = get_client()
    scored, unanswerable = [], []

    for q in queries:
        retrieved, latency_ms = retrieve_chunk_ids(client, q.query_text, k)

        if q.category == "unanswerable":
            unanswerable.append({"query": q.query_text, "retrieved": retrieved, "latency_ms": latency_ms})
            continue

        relevant = set(q.relevant_chunk_ids)
        scored.append({
            "query_id": q.query_id,
            "category": q.category,
            "precision": precision_at_k(retrieved, relevant, k),
            "recall": recall_at_k(retrieved, relevant, k),
            "latency_ms": latency_ms,
        })

    return scored, unanswerable



def aggregate_by_category(scored: list[dict]) -> dict:
    by_category: dict[str, list[dict]] = {}
    for row in scored:
        by_category.setdefault(row["category"], []).append(row)
    return {
        cat: {
            "n": len(rows),
            "mean_precision": sum(r["precision"] for r in rows) / len(rows),
            "mean_recall": sum(r["recall"] for r in rows) / len(rows),
        }
        for cat, rows in by_category.items()
    }

def write_report(scored: list[dict], unanswerable: list[dict], k: int) -> None:
    overall_p = sum(r["precision"] for r in scored) / len(scored)
    overall_r = sum(r["recall"] for r in scored) / len(scored)
    by_cat = aggregate_by_category(scored)

    lines = [
        "# Arc A — Tier 1 Baseline Report",
        "",
        f"**Overall (k={k}):** precision@{k} = {overall_p:.2f}, recall@{k} = {overall_r:.2f}",
        "",
        "## By category", "",
        "| Category | n | mean precision@k | mean recall@k |",
        "|---|---|---|---|",
    ]
    for cat, s in sorted(by_cat.items()):
        lines.append(f"| {cat} | {s['n']} | {s['mean_precision']:.2f} | {s['mean_recall']:.2f} |")

    lines += ["", "## Unanswerable queries (manual review, not in the numbers above)", ""]
    for row in unanswerable:
        lines.append(f"- **{row['query']}** -> retrieved {row['retrieved']}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    queries = load_queries()  # from Lesson A.4
    scored, unanswerable = evaluate(queries)
    write_report(scored, unanswerable, k=K)
    print(f"Scored {len(scored)} queries, logged {len(unanswerable)} unanswerable -> {REPORT_PATH}")