from typing import Callable
from generate_answer import GeneratedAnswer, generate_answer
from groundness import GroundednessVerdict, score_groundedness
from qdrant_setup import search, get_client
from self_rag import self_aware_retrieve
from embed_chunks import load_chunks
from golden_queries import RetrievalQuery, load_queries
from pathlib import Path

TIER3_REPORT_PATH = Path("../tier3_report.md")

_ALL_CHUNKS_BY_ID = {c.chunk_id: c.text for c in load_chunks()}

def lookup_chunk_text(chunk_id: str) -> str:
    return _ALL_CHUNKS_BY_ID[chunk_id]

def answer_and_verify(
    query_text: str,
    retrieve_fn: Callable[[str], list[str]],
    k: int = 5,
) -> tuple[GeneratedAnswer, GroundednessVerdict | None]:
    retrieved_texts = retrieve_fn(query_text)
    answer = generate_answer(query_text, retrieved_texts)

    if not answer.used_context:
        return answer, None  # nothing to verify -- the model declined

    verdict = score_groundedness(answer.answer_text, retrieved_texts)
    return answer, verdict


def tier1_retrieve(query_text: str, k: int = 5) -> list[str]:
    client = get_client()
    results = search(client, query_text, k=k)
    return [p.payload["text"] for p in results.points]


def tier2_retrieve(query_text: str, k: int = 5) -> list[str]:
    client = get_client()
    result = self_aware_retrieve(client, query_text, k=k)
    # texts aren't stored on SelfRAGResult directly -- re-fetch by the final chunk_ids
    return [lookup_chunk_text(cid) for cid in result.final_chunk_ids]

def evaluate_tier3(queries: list[RetrievalQuery], retrieve_fn, label: str) -> dict:
    answered_verdicts = []
    unanswerable_results = []

    for q in queries:
        answer, verdict = answer_and_verify(q.query_text, retrieve_fn)

        if q.category == "unanswerable":
            unanswerable_results.append({
                "query": q.query_text,
                "correctly_declined": not answer.used_context,
                "answer_text": answer.answer_text,
            })
            continue

        if answer.used_context and verdict is not None:
            answered_verdicts.append(verdict.grounded)

    hallucinated = sum(1 for v in answered_verdicts if v != "full")
    hallucination_rate = hallucinated / len(answered_verdicts) * 100 if answered_verdicts else 0.0

    correct_refusals = sum(1 for r in unanswerable_results if r["correctly_declined"])
    refusal_accuracy = correct_refusals / len(unanswerable_results) * 100 if unanswerable_results else 0.0

    return {
        "label": label,
        "hallucination_rate": hallucination_rate,
        "refusal_accuracy": refusal_accuracy,
        "n_answered": len(answered_verdicts),
        "n_unanswerable": len(unanswerable_results),
        "unanswerable_detail": unanswerable_results,
    }

def write_tier3_report(tier1: dict, tier2: dict) -> None:
    lines = [
        "# Arc A — Tier 3: Hallucination Rate, Tier 1 vs. Tier 2 Retrieval",
        "",
        "| | Hallucination rate | n answered | Refusal accuracy | n unanswerable |",
        "|---|---|---|---|---|",
        f"| Tier 1 retrieval | {tier1['hallucination_rate']:.1f}% | {tier1['n_answered']} "
        f"| {tier1['refusal_accuracy']:.1f}% | {tier1['n_unanswerable']} |",
        f"| Tier 2 retrieval | {tier2['hallucination_rate']:.1f}% | {tier2['n_answered']} "
        f"| {tier2['refusal_accuracy']:.1f}% | {tier2['n_unanswerable']} |",
        "",
        "Same generation prompt, same groundedness judge, same golden set for both rows -- "
        "only the retriever changed. `n answered` can differ between rows: the retriever "
        "affects what context the model sees, which affects whether it chooses to answer "
        "at all -- a real, expected downstream effect of the one variable that changed, "
        "not an inconsistency.",
        "",
        "## Unanswerable-query detail, both tiers",
        "",
        "| Query | Tier 1 | Tier 2 |",
        "|---|---|---|",
    ]
    tier2_by_query = {r["query"]: r["correctly_declined"] for r in tier2["unanswerable_detail"]}
    for r1 in tier1["unanswerable_detail"]:
        t1_status = "declined" if r1["correctly_declined"] else "ANSWERED"
        t2_status = "declined" if tier2_by_query.get(r1["query"]) else "ANSWERED"
        lines.append(f"| {r1['query']} | {t1_status} | {t2_status} |")

    TIER3_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    queries = load_queries()

    tier1_results = evaluate_tier3(queries, tier1_retrieve, "Tier 1 (naive retrieval)")
    tier2_results = evaluate_tier3(queries, tier2_retrieve, "Tier 2 (self-aware retrieval)")

    for r in [tier1_results, tier2_results]:
        print(f"{r['label']}: hallucination rate = {r['hallucination_rate']:.1f}% (n={r['n_answered']}), "
              f"refusal accuracy = {r['refusal_accuracy']:.1f}% (n={r['n_unanswerable']})")

    write_tier3_report(tier1_results, tier2_results)
    print(f"Tier 3 report written -> {TIER3_REPORT_PATH}")

    #tier1 = evaluate_tier3(queries, tier1_retrieve, "t1")
    #tier2 = evaluate_tier3(queries, tier2_retrieve, "t2")

    # manual per-query check, since the function doesn't expose this yet
    #for q in queries:
        #if q.category == "unanswerable":
           # continue
        #a1, _ = answer_and_verify(q.query_text, tier1_retrieve)
        #a2, _ = answer_and_verify(q.query_text, tier2_retrieve)
        #if a1.used_context != a2.used_context:
            #print(f"{q.query_id} flipped: Tier1 used_context={a1.used_context}, Tier2 used_context={a2.used_context}")
