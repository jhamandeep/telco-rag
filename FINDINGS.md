# TelcoRAG — Findings Ledger

*Source of truth for the final LaTeX paper (academic style, abstract → conclusion).
The paper is assembled at the end from (a) the `results_*.json` files and (b) this
ledger of narrative findings. Append one block per rung as we go; never retype
numbers from memory — copy them from the JSON.*

---

## Target paper skeleton
| Paper section | Drawn from |
|---|---|
| Abstract | one-paragraph summary of the progression + headline finding |
| 1. Introduction / problem | 5G RAN troubleshooting RAG; symptom-vs-spec vocabulary gap |
| 2. Method: gold set | 22 paraphrased queries, containment lint, difficulty tiers |
| 3. Method: metrics | P@K, R@K, MRR; why MRR headlines; the P@5 ceiling |
| 4. Method: corpus & pipeline | 26,203 pts; dense → rerank → rewrite; models |
| 5. Results | the master progression table + per-stage findings |
| 6. Discussion | salad tax, embedder-bound ceiling, corpus-integrity incident |
| 7. Conclusion | "local is good enough; the ceiling is the embedder" |
| Appendix | reproducibility (env, commands), per-query tables |

---

## Environment (reproducibility)
- **Host:** Windows 11, Intel Ultra 9 285K, RTX 5070 (12 GB, sm_120), 32 GB RAM.
- **Embedder:** Ollama `nomic-embed-text` (768-D). **Local generator:** Ollama `qwen2.5:7b` (temp 0 → deterministic).
- **Frontier generator:** OpenAI `gpt-5.5` via Chat Completions API (reasoning model; ignores temperature → non-deterministic).
- **Vector store:** Qdrant server (Docker, persistent volume), collection `telco_ran`, **26,203 points** = 16 troubleshooting + 26,187 spec chunks (7 ETSI 3GPP specs).
- **Reranker:** `BAAI/bge-reranker-base` cross-encoder. **torch:** 2.9.1+cu128.
- **Gold set:** `gold_set_v1.jsonl` (frozen). **Cutoffs:** K ∈ {1, 3, 5, 10}.

---

## Method: gold set (22 queries)
- 22 engineer queries, each **paraphrased** (never corpus text) so retrieval is tested on meaning, not string overlap.
- **Containment lint** C = |Q ∩ D| / |Q| over content tokens; all queries C ≤ 0.78, mean 0.40; warn line 0.85. Q05 = 0.00 (purest semantic probe).
- **Difficulty tiers:** easy 8 (key terms present), medium 8 (symptom described), hard 6 (downstream effects only).
- **4 multi-gold** queries (Q02, Q07, Q09, Q11) keep Recall non-degenerate.
- Answer key uses semantic labels (`TS-xx`), never storage point ids — survives re-ingest.

## Method: metrics
- **Precision@K** = gold-in-topK / K. Ceiling = min(G,K)/K; 1-gold query caps at 0.20 → **P@5 is not a headline metric here** (18/22 queries are 1-gold).
- **Recall@K** = distinct gold found / total gold. Completeness.
- **MRR** = mean(1 / rank of first gold). Order-sensitive → **headline metric**.
- Correctness = label membership (`hit.id ∈ gold.relevant`), NOT a score threshold. Binary relevance.
- Key mechanical fact used throughout: **reranking reorders a fixed fetched set → R@10 invariant; only re-fetching (rewrite) can change R@10.**

---

## MASTER RESULTS TABLE (overall, n=22)
| Stage | file | MRR | R@5 | R@10 | P@5 |
|---|---|---|---|---|---|
| Dense (baseline) | results_baseline_20260703_204931 | 0.770 | 0.795 | 0.841 | 0.200 |
| + Rerank | results_rerank_20260703_220513 | 0.780 | 0.818 | 0.841 | 0.200 |
| + Rewrite (qwen2.5:7b) | results_rewrite_20260704_094720 | 0.848 | 0.909 | 0.909 | 0.218 |
| + Rewrite (gpt-5.5) run A | results_rewrite_20260704_101208 | 0.864 | 0.909 | 0.909 | 0.218 |
| + Rewrite (gpt-5.5) run B | results_rewrite_20260704_102745 | 0.848 | 0.909 | 0.909 | 0.218 |

*gpt-5.5 ignores temperature -> non-deterministic. Two runs gave MRR 0.864 and 0.848
(spread 0.016). qwen (temp 0) = 0.848 exactly. The two models are indistinguishable
within gpt-5.5's own run-to-run noise -> a local 7B matches a frontier model here.
In every rewrite run, hard R@10 = 0.750 and Q05 fails -> that ceiling is robust.*

### Hard tier (n=6) — where all the action is
| Stage | MRR | R@5 | R@10 |
|---|---|---|---|
| Dense | 0.352 | 0.333 | 0.500 |
| + Rerank | 0.306 | 0.417 | 0.500 |
| + Rewrite (qwen) | 0.639 | 0.750 | 0.750 |
| + Rewrite (gpt-5.5) | 0.611 | 0.750 | 0.750 |

---

## Stage findings

### Baseline (dense retrieval)
- Healthy MRR 0.770. **Tier gradient monotonic** (easy 0.938 > medium 0.917 > hard 0.352) → the gold set discriminates; the hard tier is the real test.
- P@5 = 0.200 is the mathematical ceiling (confirmed: medium P@5 = 0.275 because it holds the multi-gold queries).

### Rerank (bge-reranker-base)
- Small net gain (MRR +0.010, R@5 +0.023). **Easy MRR 0.938 → 1.000** (clean win where the cross-encoder is confident, scores ≈0.99).
- **Hard tier splits: R@5 +0.083 but MRR −0.046.** The reranker's scores collapse to ≈0.000 on hard queries (can't discriminate), so its reorder is noise that can demote a lucky dense hit.
- **R@10 invariant everywhere** (+0.000) — confirms the reorder-a-fixed-set mechanism.
- *Lesson: track both R@5 and MRR — a single metric hides the split.*

### Rewrite + rerank (qwen2.5:7b, local)  ← the breakthrough
- Overall **MRR 0.848, R@5 0.909, R@10 0.909**; lift over rerank: MRR +0.068, R@5 +0.091, **R@10 +0.068** (R@10 moved — re-fetch works).
- **Hard tier transformed:** MRR 0.306→0.639, R@5 0.417→0.750, **R@10 0.500→0.750**.
- **Rescues (visible via `(rw)` tag):** Q14→TS-09 and Q19→TS-12, both from R@10=0 to rank-1 `(rw)`. Textbook: the clean rewrite moved the query vector onto the gold and re-fetched it.
- **Salad tax (cost, also visible):** Q05→TS-01 still fails (rewrite was a term-dump incl. wrong "too-early"); Q11 lost a gold (TS-06 evicted, R@10 1.0→0.5); Q15 dropped a rank (rewrite-fetched chunk outranked gold) → easy MRR −0.062.

### Rewrite + rerank (gpt-5.5, frontier)
- Two samples (non-deterministic): MRR **0.864** and **0.848**. qwen (deterministic) = 0.848.
  **The models are indistinguishable within gpt-5.5's own run-to-run spread** — the earlier
  "frontier wins by 0.016" reading did not survive a second sample.
- gpt-5.5's rewrites are reliably *clean* (no salad), but that cleanliness did **not** buy a
  meaningful retrieval win, because the remaining failures are not generator-limited.
- Rewrite quality varies which easy query it dents: run A dinged Q15, run B dinged Q08
  (a rewrite-fetched spec chunk outranked the gold) — easy MRR 1.000 vs 0.938 across runs.
- **KEY FINDING — Q05 confirmed embedder-bound.** With gpt-5.5's clean, correct rewrite,
  TS-01 is *absent from Q05's top 10 entirely* (not buried — never fetched, by either the
  original or the rewritten query). Neither query embeds near TS-01 under nomic-embed-text.
  ⇒ The hard ceiling is the **embedder**, not the generator or the reranker. A perfect LLM
  rewrite cannot rescue what the embedder won't place near the answer. Fix is architectural:
  **hybrid retrieval (BM25 + dense)** or a stronger embedder.
- Caveat: single samples are unreliable for gpt-5.5; report the spread, not one number.

---

## Methodology incident (belongs in Discussion — a genuine lesson)
Between runs, a system/Docker reset wiped the Qdrant server; the app fell back to
embedded `QDRANT_MODE=local` (16 points, **spec lane = 0**). Metrics *inflated* to a
fake MRR 0.930 / R@10 1.000 — a regression disguised as a breakthrough (top-10 of
~16 almost always contains the gold).
- **Caught by:** the lift table's guard-rail — R@10 *changed* under reranking, which is *impossible* if the candidate set is fixed ⇒ the corpus must have changed. Confirmed with `check_corpus.py`.
- **Fixed by:** `bootstrap.py` (pin `QDRANT_MODE=server`, persistent Docker volume, skip-ingest-when-complete). Deterministic ingest reproduced the pre-wipe numbers to 3 decimals → corpus verified whole.
- **Lesson for the paper:** an eval is only comparable if BOTH the gold set AND the corpus are frozen. We froze the gold set; the corpus taught us to freeze it too.

---

## Key insights (for Discussion / Conclusion)
1. **MRR + R@K over P@K** for a mostly-1-gold corpus (P@5 ceiling = 0.20).
2. **Reranking helps only where the cross-encoder is confident;** on genuinely hard queries it adds noise. It cannot raise R@10 (fixed set).
3. **Query rewriting is the lever that moves R@10** — re-fetch changes the candidate set. Hard R@10 0.500 → 0.750.
4. **Clean rewrites rescue; salad rewrites cost.** Failure mode = term-dumping / query drift (visible in the `(rw)` tags).
5. **Local ≈ frontier for this task.** qwen2.5:7b within ~0.016 MRR of gpt-5.5; the frontier edge is *avoiding self-inflicted noise*, recoverable for free via a tighter prompt.
6. **The residual hard ceiling (Q05) is embedder-bound, not model-bound** — the single most report-worthy finding. Points to hybrid retrieval (BM25 + dense) or a stronger embedder, not a bigger LLM.

---

## Results file inventory
| file | run | generator |
|---|---|---|
| results_baseline_20260703_204931.json | plain dense | — |
| results_rerank_20260703_220513.json | dense + rerank | — |
| results_rewrite_20260704_094720.json | dense + rewrite + rerank | ollama:qwen2.5:7b |
| results_rewrite_20260704_101208.json | dense + rewrite + rerank | openai:gpt-5.5 |
*(post-wipe reruns 090751 / 100038 reproduce the above and can be ignored.)*

---

## Open threads / remaining rungs
- [x] **Q05 diagnosed (both runs): embedder-bound.** TS-01 absent from the merged pool even
  with gpt-5.5's clean rewrite ⇒ nomic-embed-text won't place it near the query. Not a rerank
  or generator fix.
- [ ] **Rung 7 (next) — hybrid retrieval (BM25 + dense fusion).** Directly attacks the embedder
  ceiling: lexical BM25 catches exact-term overlap ("A3 offset", "time-to-trigger", "handover")
  that dense dilutes. Test whether it finally pulls TS-01 into Q05's pool (hard R@10 > 0.750).
- [ ] Tighten the qwen rewrite prompt to kill the salad; measure the (small) lift.
- [ ] RAGAS generation eval (faithfulness + answer relevancy), qwen vs gpt-5.5.
- [ ] **Rung 8 — assemble the LaTeX paper** from this ledger + the JSON.
- [ ] (optional) `compare_runs.py` to auto-tabulate all `results_*.json` for the paper.
