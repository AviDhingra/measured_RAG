# Self-Aware, Verified RAG over Real AI-Agent Security Incidents

A three-tier retrieval system, built and measured one honest layer at a time: a naive baseline, a self-critic that catches its own uncertain retrievals, and an independent groundedness check on every generated answer — over a real, dated corpus of AI-agent security incidents (EchoLeak, the GitHub MCP prompt-injection campaign, s1ngularity, and dozens more).

**The headline finding:** the self-critic measurably improved retrieval on the corpus's hardest category (ambiguous questions with more than one valid answer). It did **not** clearly improve the trustworthiness of the final generated answer — refusal accuracy on genuinely unanswerable questions stayed flat, and the hallucination-rate comparison itself proved unstable across repeated runs, even with generation and judging pinned to `temperature=0`. Better retrieval did not automatically mean more trustworthy generation. That's the real result, reported as found, not smoothed into a cleaner story.

---

## The three-tier story

1. **Tier 1 — baseline.** Chunk the corpus, embed locally, index in Qdrant, retrieve. Measured against a hand-built 20-query golden set spanning five failure modes: direct lookup, paraphrase, multi-hop, unanswerable, and ambiguous.
2. **Tier 2 — self-critic.** After retrieving, an LLM judge scores whether the retrieved chunks are actually sufficient to answer the query. If not, it reformulates the query and retries — bounded to two attempts. A scoped, measured implementation of the Self-RAG / Adaptive-RAG research direction.
3. **Tier 3 — groundedness.** Closes the retrieve-then-generate loop for the first time. A grounded generation prompt, then an independent verifier scoring every answer full / partial / none, plus a refusal-accuracy check on the genuinely unanswerable questions.

---

## Inspiration

Tier 2's retrieve-critique-retry loop borrows its core idea from two lines of research, adapted rather than reproduced:

- **Self-RAG** ([Asai et al., 2024](https://arxiv.org/abs/2310.11511), ICLR) trains a language model end-to-end to emit special "reflection tokens" that decide when to retrieve and critique its own retrieved passages and generations — the decision is baked directly into next-token prediction.
- **Adaptive-RAG** ([Jeong et al., 2024](https://arxiv.org/abs/2403.14403), NAACL) trains a separate classifier to predict a query's complexity upfront, then routes it to one of several retrieval strategies *before* any retrieval happens.

This project does neither, on purpose. No model is trained, and no upfront classifier decides anything before retrieval starts. Instead, an already-capable frontier model (Claude Haiku) is prompted, *after* retrieval, to judge whether what came back is actually sufficient — and retries with a reformulated query if not, bounded to two attempts. It borrows the core idea both papers share — a system that knows when its own retrieval isn't good enough — without needing training infrastructure or labeled data, at the honest cost of being less principled than either paper's actual method.

---

## Results

### Tier 1 — the baseline

Overall: **precision@5 = 0.24, recall@5 = 0.75**

| Category | n | precision@5 | recall@5 |
|---|---|---|---|
| ambiguous | 4 | 0.10 | 0.25 |
| direct_lookup | 4 | 0.25 | 1.00 |
| multi_hop | 4 | 0.35 | 0.88 |
| paraphrase | 4 | 0.25 | 0.88 |

`direct_lookup` recall of 1.00 means the naive baseline found every correct chunk that existed for the easy questions — precision looks lower only because most direct-lookup questions only have 1–2 correct chunks against a top-5 window. `ambiguous` is the real weak spot: both precision and recall trail every other category by a wide margin, on questions that genuinely have more than one defensible answer.

### Tier 2 — self-critic, vs. Tier 1

| Category | Tier 1 precision | Tier 2 precision | Tier 1 recall | Tier 2 recall |
|---|---|---|---|---|
| ambiguous | 0.10 | 0.15 | 0.25 | 0.38 |
| direct_lookup | 0.25 | 0.25 | 1.00 | 1.00 |
| multi_hop | 0.35 | 0.35–0.40 | 0.88 | 0.88–1.00 |
| paraphrase | 0.25 | 0.25 | 0.88 | 0.88 |

The self-critic improved exactly the category Tier 1 identified as weak, and left the already-easy categories untouched — the behavior you'd want from a well-targeted fix, not a blanket "try harder everywhere" change. `multi_hop` showed some run-to-run variance even at `temperature=0` (see note below); the other three categories were stable across repeated runs.

**Cost:** ~19% of queries needed at least one retry. Average 1.25 attempts per query. Average added latency ~3.3–3.7 seconds per query.

### Tier 3 — hallucination rate and refusal accuracy, Tier 1 vs. Tier 2 retrieval

| | Refusal accuracy |
|---|---|
| Tier 1 retrieval | 25% (1 of 4 unanswerable questions correctly declined) |
| Tier 2 retrieval | 25% (same) |

Refusal accuracy was stable and identical across every run — self-aware retrieval did not improve the system's ability to recognize genuinely unanswerable questions at the generation layer.

**Hallucination rate did not stabilize.** Across repeated runs at `temperature=0`, results ranged from Tier 1 = 13.3–20.0% and Tier 2 = 6.7–18.8% — the direction of the comparison (which tier hallucinated less) reversed between runs on identical code. Anthropic's API does not guarantee determinism even at `temperature=0` (no seed parameter is exposed; best-effort only). At this sample size (n≈15 answered queries per run), that residual noise is large enough to flip the headline conclusion. Rather than report one run as "the" number, this is reported honestly as a range with the instability itself named as a finding.

---

## Setup and running it yourself

```bash
# 1. Qdrant, local
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 2. Ollama, for local embeddings (adjust host/port to your own setup)
ollama pull nomic-embed-text

# 3. Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."   # or set it as a persistent env var

# 4. Python deps
pip install -r requirements.txt

# 5. Run the pipeline, in order
python src/build_corpus.py
python src/embed_chunks.py
python src/qdrant_setup.py
python src/evaluate_tier1.py
python src/evaluate_tier2.py
python src/evaluate_tier3.py
```

`golden_queries.jsonl` is checked in — you don't need to rebuild the golden set to reproduce these numbers, only to extend it.

## Corpus and credits

Source incident data: [`awesome-ai-agent-incidents`](https://github.com/h5i-dev/awesome-ai-agent-incidents), MIT licensed. This is a living document; `incidents.meta.json` records the exact snapshot this project was built and measured against, since the source content can change over time.

## What I'd do differently with more time

- **Rank-aware retrieval metrics** (NDCG, mean reciprocal rank) instead of unordered precision@k — the current metric treats a hit at rank 1 the same as a hit at rank 5.
- **A larger golden set**, specifically more than 4 queries per category — at this sample size, one query changing outcome moves an entire category's average, which is the single biggest limitation in every result above.
- **Sentence-level claim decomposition** for groundedness scoring, instead of one holistic full/partial/none judgment per answer — would make the "partial" category, already flagged as the most dangerous failure mode, measurable in more detail.
- **Report every noisy metric as a range across multiple runs by default**, not just as a follow-up when a number looks surprising — Tier 3's instability was found by accident, through repeated re-runs; it should have been the default methodology from the start.
