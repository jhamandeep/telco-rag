# TelcoRAG

**A cost-adaptive agentic RAG system for 5G RAN mobility troubleshooting — built from scratch, measured end-to-end, and shipped as a working product on a single workstation.**

> 📄 **Read the full paper: [`TelcoRAG_Report.pdf`](TelcoRAG_Report.pdf)** — a 20-page report that documents the design, the measured retrieval progression, four architecture views, the RAGAS evaluation, the real-world hardening, and the production economics. Every number in this README is drawn from it and is reproducible with the eval harness below.

*IIT Bombay — Generative AI, final project · Author: Mandeep Kumar Jha · 2026.*

---

## Table of contents
- [The problem](#the-problem)
- [What TelcoRAG does](#what-telcorag-does)
- [Key results](#key-results-measured)
- [How it works](#how-it-works)
- [The product (console)](#the-product-console)
- [Real-world hardening](#real-world-hardening)
- [Production economics](#production-economics-why-it-scales)
- [Quickstart](#quickstart-powershell)
- [Reproduce the results](#reproduce-the-results)
- [Repository structure](#repository-structure)
- [Configuration & privacy](#configuration--privacy)
- [Hardware & stack](#hardware--stack)

---

## The problem

A 5G RAN engineer diagnosing a mobility fault does not begin with a specification
clause number. They begin with a **symptom**: *"handsets drop mid-handover on the
motorway," "battery drains overnight near a cell border," "a strong carrier is invisible
to idle phones."* The authoritative answer, however, lives in 3GPP specifications
written in an entirely different vocabulary — parameter names (`A3-offset`,
`timeToTrigger`, `T310`, `SnonIntraSearchP`), timers, and RRC procedures.

There is a **vocabulary gap** between how a fault is *observed* and how its remedy is
*documented*. Keyword search over specifications fails because the engineer's words are
not the specification's words. A generic chatbot fails because it was never trained on
this operator's scenario library and will confidently hallucinate parameter values.
Closing that gap — reliably placing the right clause and the right troubleshooting
scenario in front of the model for a query phrased as a *symptom* — is exactly what
TelcoRAG does.

## What TelcoRAG does

TelcoRAG is **agentic RAG**: instead of a single retriever and a fixed pipeline, it uses
**multiple retrieval lanes** (dense semantic + lexical BM25 + LLM query rewrite) and a
**control loop that decides, per query, how much work to do** — escalating from cheap to
expensive only when a cheaper step proves insufficient. An engineer asks in plain
language; TelcoRAG returns a **grounded, cited** answer, plus a visible trace of *how* it
retrieved the answer and *what it cost*.

The whole system runs **offline on one workstation**. A frontier model (e.g. GPT-class)
is an **optional accelerant** for synthesis-heavy queries — never a per-query dependency.
Commercially sensitive specifications and uploaded operator documents never leave the
machine.

## Key results (measured)

Scored on a **frozen 22-query gold set** (easy / medium / hard tiers), using
Precision@K, Recall@K, and Mean Reciprocal Rank (MRR):

| Stage | MRR | R@10 | LLM/query | Note |
|---|---|---|---|---|
| Dense retrieval | 0.770 | 0.841 | 0 | semantic baseline |
| + Cross-encoder rerank | 0.780 | 0.841 | 0 | reorders top-k |
| + LLM query rewrite | 0.848 | 0.909 | 1.0 | moves the query vector |
| + BM25 / RRF hybrid | 0.811 | **0.932** | 0 | lexical lane |
| **All four lanes fused (static max)** | **1.000** | **1.000** | 1.0 | the ceiling |
| **ReAct agent (adaptive)** | **0.936** | 0.977 | **0.14** | recovers the ceiling, cheaply |

- **The agent recovers ~94% of the static ceiling at 14% of the LLM cost.** 19 of 22
  queries are answered with **zero** LLM calls; only 3 reach the rewrite lane.
- **A local 7B model matches a frontier model within noise.** On the rewrite stage,
  `qwen2.5:7b` scores **0.848** vs `gpt-5.5` **0.864** — and the local model is actually
  *higher* on the hard tier (0.639 vs 0.611).
- **Generation is grounded.** Gold-free **RAGAS** faithfulness is **0.917**
  (self-grading upper bound); **0.958** on an independent judge.
- **A diagnostic finding drives the architecture.** The hardest query (Q05) is
  *embedder-bound*, not model-bound — no rewrite places its gold under dense retrieval —
  and it is cracked only when the BM25-on-rewrite lane joins the fusion. The
  cross-encoder reranker collapsed (a gate fired) on **10 of 22** queries, which is why
  the agent *distrusts* it structurally.

## How it works

**Four retrieval lanes.** Every query can be answered from the cross-product of two
encoders and two query forms: `do` (dense, original), `dr` (dense, rewritten),
`bo` (BM25, original), `br` (BM25, rewritten). Dense captures *meaning*; BM25 captures
*exact terms* the dense space dilutes. They fix **different** queries, so fusing them
beats any single lane.

**RRF fusion.** Lanes are combined by **Reciprocal Rank Fusion** (rank-space, not
score-space; `k=60`), so a document that ranks well in *either* lane surfaces — the
mechanism that lifts hard-tier recall.

**The ReAct agent.** A strict-budget escalation loop: dense first; if a structural
sufficiency critic isn't satisfied, add the BM25 lane and fuse; only if still
insufficient, spend one LLM call to rewrite the query and refetch both lanes. A
**gated reranker** overrules the cross-encoder when it collapses (keeps the RRF order).
This is why the agent reaches the ceiling on hard queries while paying, on average,
**0.14 LLM calls per query**.

**Conversation memory (history-aware retrieval).** A real troubleshooting session is
multi-turn. A follow-up like *"what parameters for the same?"* is meaningless to a
stateless retriever, so TelcoRAG **condenses** the follow-up into a standalone query
using the recent turns before retrieving, and shows the rewrite to the user as an
*"Understood as"* chip. The prior turns are also passed to the generator.

## The product (console)

A single-file **production console** (`ui/console.html`) over a FastAPI engine
(`serve.py`, 16 endpoints):

- **Grounded, cited answers** with an *"Not covered by the sources"* honesty boundary.
- A **live retrieval trace** (dense → BM25 fusion → rewrite) with the **per-query cost**
  (retrieval calls vs LLM calls).
- **Conversation memory** with a multi-conversation history sidebar.
- A **live-editable corpus**: upload PDFs/Word/txt and delete them (chunk counts update).
- **Grounded artifact generation**: report / table / Mermaid diagram / mind-map / study
  guide / quiz — generated over the retrieved context (synthesis auto-routes to the
  frontier model), with a client-side Mermaid sanitizer so diagrams always render.
- Apple-grade interface, light/dark, resizable panels.

## Real-world hardening

A static benchmark scored retrieval at MRR 1.000 — but deploying the console and running
a **real 5-turn conversation** surfaced five failures a benchmark can't see. Each was
diagnosed, fixed, and locked behind a regression harness (`test_improvements.py`) that
now passes **10/10**:

| Failure found in real use | Fix shipped |
|---|---|
| Refuses despite usable context | partial-answer prompt (answer + list gaps) |
| Fabricates fake vendor CLI from specs | no-command guard (name parameters instead) |
| Relays wrong web commands uncritically | unverified-web-source flag |
| Follow-up query malformed | condenser hardening |
| No scenario for idle-mode reselection | **3 new scenarios authored + ingested** |

The deepest fix was **data, not model**: three idle-reselection scenarios
(`SnonIntraSearch`, inter-frequency priority, `Qrxlevmin`) — the same lesson as the Q05
case study. Synthesis-heavy and command queries now **auto-route to the frontier model**,
where the local 7B previously refused or hallucinated.

## Production economics (why it scales)

The engineering thesis is **quality per LLM call, not per query**:

- Pure-retrieval methods (rerank, BM25, RRF) add quality at **zero** LLM cost.
- The agent issues **~140 LLM calls per 1000 queries** vs 1000 for a naive per-query
  pipeline — bounded, predictable cost.
- **Data sovereignty:** the full stack (embedder, generator, vector store) runs offline;
  the frontier model is optional and falls back to local on any failure.
- **Horizontal scale:** the FastAPI sidecar is stateless (memory lives client-side), so
  it replicates behind a load balancer over one shared Qdrant.

---

## Quickstart (PowerShell)

```powershell
# 1. Models (Ollama, native GPU)
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

# 2. Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env         # then edit .env (OPENAI_API_KEY is optional)

# 3. Add the 3GPP spec PDFs into .\data\ (gitignored; download your own copies):
#    38.331, 38.300, 38.321, 38.133, 38.215, 38.423, 38.413

# 4. Bring everything up + verify persistence (Docker Desktop running)
python bootstrap.py

# 5. Run the product
uvicorn serve:app --port 8000                      # engine (terminal 1)
cd ui; python -m http.server 5173                  # UI (terminal 2)
#    open http://localhost:5173/console.html
```

## Reproduce the results

```powershell
python eval\run_eval.py                 # dense baseline           -> MRR 0.770
python eval\run_eval.py --rerank        # + cross-encoder          -> MRR 0.780
python eval\run_eval.py --rewrite       # + LLM query rewrite      -> MRR 0.848
python eval\run_eval.py --hybrid        # + BM25 / RRF             -> MRR 0.811, R@10 0.932
python eval\run_eval.py --hybrid --rewrite   # all four lanes      -> MRR 1.000
python eval\run_agent.py                # the adaptive ReAct agent -> MRR 0.936 @ 0.14 LLM/q
python eval\run_ragas.py                # gold-free generation eval (faithfulness 0.917)
python test_improvements.py             # conversational regression harness (10/10)
```

Each run saves a config-stamped `results_*.json` so the numbers are auditable.

## Repository structure

| Path | Role |
|---|---|
| `serve.py` | FastAPI engine — the agent exposed as a product API (16 endpoints) |
| `ui/console.html` | the production console (single file) |
| `agent.py` | the ReAct escalation agent (the control loop) |
| `search.py`, `rerank.py`, `rewrite.py`, `bm25_tool.py`, `eval/rrf.py` | the retrieval lanes and fusion |
| `eval/` | the measured eval harness (`run_eval`, `run_agent`, `run_ragas`, `metrics`, `gold_set_v1`) |
| `corpus/troubleshooting.jsonl` | 19 hand-authored troubleshooting scenarios |
| `chunker.py`, `embed.py`, `ingest_specs.py`, `ingest_troubleshooting.py` | the ingest pipeline |
| `bootstrap.py` | one-command bring-up with persistence verification + snapshots |
| `test_improvements.py` | conversational regression harness |
| `config.py`, `.env.example`, `docker-compose.yml` | configuration and the Qdrant server |
| **`TelcoRAG_Report.pdf`** | **the full 20-page report** |

## Configuration & privacy

All settings live in `config.py`, read from `.env`. **`.env` is gitignored — never commit
it.** With no `OPENAI_API_KEY`, generation and artifacts fall back to the local model.
The 3GPP spec PDFs under `data/` are gitignored (copyrighted; supply your own). Specs and
uploads never leave the machine.

## Hardware & stack

One Windows 11 workstation · Intel Core Ultra 9 285K · **NVIDIA RTX 5070 (12 GB)** ·
32 GB RAM · Ollama (`nomic-embed-text` 768-d embedder, `qwen2.5:7b` generator) ·
Qdrant (Docker) · cross-encoder `BAAI/bge-reranker-base` · optional OpenAI frontier model.

---

*Built from scratch as an IIT Bombay Generative AI capstone. See [`TelcoRAG_Report.pdf`](TelcoRAG_Report.pdf) for the complete write-up.*
