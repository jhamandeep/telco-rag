# TelcoRAG — Project Log & Handoff

*Paste this whole file into a new chat to resume. Also keep FINDINGS.md (the
results ledger) and the results_*.json files — together they are the full record.*

## What this is
Agentic RAG for **5G RAN / mobility troubleshooting** (IIT Bombay GenAI project).
Ingests 3GPP specs + hand-written troubleshooting scenarios, retrieves with
citations, evaluated with retrieval metrics. Runs **fully local** on one workstation,
with an optional OpenAI generator for comparison. The standout signal is the
**evaluation harness** — measured proof each stage beats the last.

## Environment
- **Windows 11**, Intel Ultra 9 285K, **RTX 5070 (12 GB, sm_120)**, 32 GB RAM, PowerShell.
- **Ollama** (native, GPU): `nomic-embed-text` (768-D) + `qwen2.5:7b` (local generator, temp 0 = deterministic).
- **Qdrant** in Docker, **server mode**, PERSISTENT volume (`./qdrant_storage`), collection `telco_ran`, http://localhost:6333.
- **torch 2.9.1+cu128** (Blackwell/sm_120). **Reranker:** `BAAI/bge-reranker-base`.
- **OpenAI (optional):** `gpt-5.5` via Chat Completions; keys in `.env` (`OPENAI_API_KEY`, `OPENAI_MODEL`); `REWRITE_BACKEND` env switches local/openai.
- **Bring-up:** `python bootstrap.py` (pins QDRANT_MODE=server, starts Qdrant, checks Ollama, skips ingest when corpus complete). `--with-generator` pulls qwen; `--force-ingest` re-ingests.
- **GOTCHA (already bit us):** if QDRANT_MODE falls to `local`, you read a 16-point embedded store, not the 26k server. `python eval\check_corpus.py` guards this. Persistent volume now prevents the wipe.

## How we work (operating rules)
- **Step-wise**: one rung, a check (OK) at each, understanding tested before advancing.
- Depth over coverage; a worked number per formula; a visual per concept; ~25-word sentences.
- Don't add a dependency/element without saying what it is and why it's here now.
- When a script works standalone but fails on import, the traceback's FILE PATH is the truth (import shadowing).

## Progress map
1. OK Foundation — embeddings + vector store
2. OK Ingest — troubleshooting + spec lanes (26,203 points)
3. OK Dense retrieval — search.py
4. OK **Eval harness** — gold_set_v1 (22 queries), P@K/R@K/MRR, baseline recorded
5. OK **Reranking** — bge-reranker cross-encoder; measured lift
6. OK **Query rewrite** (single hop) — qwen/gpt-5.5 rewrite → merge → rerank; measured lift
7. -> **AGENTIC LOOP (NEXT)** — ReAct (Thought/Action/Observation) + reflection critic + multi-hop, with tools {dense, hybrid/BM25, rewrite}. THIS is the real agentic RAG; everything above was its measured baseline.
8. .. Generation eval — RAGAS (faithfulness + answer relevancy)
9. .. Report — LaTeX academic paper (abstract->conclusion), PDF, from FINDINGS.md + JSON

## Results so far (details in FINDINGS.md; numbers in results_*.json)
Overall MRR: dense **0.770** -> +rerank **0.780** -> +rewrite **0.848** (qwen) / 0.848-0.864 (gpt-5.5).
Hard tier R@10: dense 0.500 -> rerank 0.500 (reranking can't refetch) -> rewrite **0.750** (refetch works).
Key findings:
- MRR/R@K headline, not P@5 (ceiling 0.20 on 1-gold queries).
- Reranking helps only where the cross-encoder is confident; can't move R@10 (fixed set).
- Query rewrite moves R@10 (refetch changes the candidate set). Q14, Q19 rescued.
- Clean rewrites rescue; salad rewrites cost. Visible via the (rw) provenance tag.
- **Local qwen == frontier gpt-5.5 within noise** (0.848 vs 0.848-0.864).
- **Q05 is embedder-bound**: even gpt-5.5's clean rewrite doesn't place TS-01 in the top 10. Fix is a LEXICAL/hybrid tool, not a bigger LLM -> motivates BM25 as an agent tool.

## Files (in `telco-rag\`)
Root: config.py, store.py, embed.py, chunker.py, extract.py, ingest_troubleshooting.py, ingest_specs.py, search.py, smoke_test.py, docker-compose.yml, requirements.txt, **bootstrap.py**, **FINDINGS.md**, HANDOFF.md.
`eval\`: **gold_set_v1.jsonl**, validate_gold.py, **metrics.py**, **run_eval.py** (modes: default / --rerank / --rewrite; --show; --k), **rerank.py**, **rewrite.py** (backend local/openai), **check_corpus.py**, **test_openai.py**.
Data: `data\*.pdf` (7 ETSI specs, gitignored), `qdrant_storage\` (persistent, gitignored).

## Mastered concepts (don't re-teach)
- Embedding = text -> 768 numbers; cosine = angle; same model for corpus AND query.
- Metrics: P@K = gold-in-K/K (ceiling min(G,K)/K); R@K = distinct gold found / total; MRR = mean(1/rank of first gold). Correctness = label membership, not score threshold.
- Reranking reorders a FIXED fetched set -> R@10 invariant. Only re-fetching (rewrite/agent) changes R@10.
- Bi-encoder (dense, fast, separate) vs cross-encoder (rerank, sharp, joint).
- Query rewrite moves the QUERY vector to a new neighbourhood -> new nearest-K.
- Eval integrity: freeze BOTH the gold set AND the corpus, or comparisons are meaningless.

## NEXT — build the agentic loop (rung 7)
Build a ReAct + reflection agentic retriever, measured on the SAME 22 gold queries so the
agentic-vs-pipeline lift is a clean delta against results_rewrite_*.json.
1. **Tools**: dense_search, hybrid_search (BM25 + dense via Reciprocal Rank Fusion), rewrite_query.
2. **Loop**: Thought (LLM decides action) -> Action (call a tool) -> Observation (docs) -> Reflect (LLM: sufficient? which gold-ish signal?) -> loop or stop. Budget: max N steps.
3. **Target**: crack Q05 (embedder-bound) by letting the agent choose the hybrid/BM25 tool; push hard R@10 past 0.750.
4. Keep the step-wise style; measure the lift; log to FINDINGS.md.

**Resume prompt:** *"Resume TelcoRAG — build the agentic RAG loop (rung 7): a ReAct + reflection retriever with tools {dense, hybrid/BM25 via RRF, rewrite}, measured on gold_set_v1 against the rewrite baseline (MRR 0.848, hard R@10 0.750). Start with the BM25/hybrid tool since Q05 is embedder-bound. Keep the step-wise style: one rung, a check, test my understanding before advancing."*
