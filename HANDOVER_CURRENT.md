# TelcoRAG — Current Handover (Product-Build Phase)

*Paste this whole file into a new chat to resume. It supersedes the older
HANDOFF.md for the product/UI phase; HANDOFF.md + FINDINGS.md remain the record
for the retrieval-engine phase (rungs 1–7).*

---

## Where we are in one line

The **retrieval engine is done and measured** (rung 7d agent: MRR 0.936, R@5 0.977,
86% fewer LLM calls than static). The **product/console is feature-complete and
polished**. The **graded deliverable — the paper/report — is NOT written yet.**
That is the single most important pending item. Deadline **12 July 2026**.

---

## THE GRADED DELIVERABLE (read this first)

- The IIT Bombay GenAI course is graded on a **written report + a 5–10 min video**
  (video mandatory). **Code is optional/supporting**, not the graded artifact.
- **The report does not exist yet.** Everything built so far is the *evidence* for
  it, not the report itself.
- The paper is ~90% supported by work already done — the measured results, the
  architecture, the six-stage retrieval progression, the agent, the product. It
  needs to be *written up*, not re-researched.
- **Recommended report structure** (end-to-end, comprehensive — the user explicitly
  wants the full arc, not just a results section):
  1. **Problem** — 5G RAN mobility troubleshooting; why RAG; why agentic.
  2. **Data** — 3GPP/ETSI spec ingest + 16 hand-written troubleshooting scenarios;
     26,203 Qdrant points; chunking strategy.
  3. **Architecture** — single-workstation local stack (Ollama, Qdrant, reranker),
     with the architecture diagram; optional OpenAI generator for comparison.
  4. **Retrieval progression (the spine)** — measured, stage by stage: dense →
     +rerank → +rewrite → hybrid (BM25+RRF) → agent. Each stage beats the last,
     with the numbers from FINDINGS.md / results_*.json. THIS is the intellectual
     core — the "measured proof each rung beats the last."
  5. **The agent (rung 7d)** — ReAct escalation loop, structural reflection critic
     (never reads CE score — corroboration across lanes / RRF margin), gated rerank
     finalize. Headline: **0.936 MRR, R@5 0.977, hard R@10 1.000, 0.14 LLM
     calls/query vs 1.0 static (86% cut).** Quality-per-tool-call is the thesis.
  6. **Product** — the console as the live demo: five patterns, deep web research,
     studio artifacts, dual runtime. The console is the video centerpiece.
  7. **Analysis / limits** — gold is a dev-time ruler only (system uses ZERO gold
     labels at inference); P@K capped by 1-gold ceiling so MRR/R@10 are the headline;
     confidence gate overrules a collapsed reranker; RAGAS (rung 8) as future work.
- Include **code snippets + diagrams** throughout (user wants it comprehensive).
- Skills available for authoring: `/mnt/skills/public/docx` (Word report),
  `/mnt/skills/public/pdf`, `/mnt/skills/public/pptx` (if slides for the video).

---

## What was built THIS phase (product + fixes)

### serve.py — the FastAPI sidecar (13 endpoints)
`/health`, `/ask` (agent + grounded cited answer), `/sources`, `/upload` (text),
**`/upload_file`** (PDF via PyMuPDF + DOCX via python-docx + txt/md, base64,
server-side extraction), `/artifact` (6 kinds: report/table/diagram/mindmap/
studyguide/quiz), `/patterns` + `/pattern` (runs the 5 lab patterns), **`/research`**
(deep web research — DuckDuckGo scoped to a telco allow-list, fetch+clean via bs4,
ingest into Qdrant, answer over enriched corpus), `/config` GET+POST (runtime
toggle local↔openai), **`/diag/openai`** + **`/diag/local`** (diagnostics).
All generation routes through `_gen_text()` respecting `_use_openai()` (toggle AND key).

### ui/console.html — the polished demo surface (~60 KB, single file)
Innova/Verizon enterprise design (dark header, "by Innova / powered by Claude",
telco-blue accent, JetBrains Mono for data). Features:
- **Resizable 3 columns** (drag dividers, widths persist to localStorage).
- **Conversation history + memory** (left sidebar, persists, click to reload thread,
  × to delete, New button) — reasoning is stored and replayed per turn.
- **Strategy pills**: Agent (7d), 🌐 Deep Research, + 5 patterns, each with its own
  trace renderer.
- **Inline collapsible "Reasoning"** per answer (Claude/OpenAI style) — shows the
  think→retrieve→reflect chain with green/amber stop-vs-escalate dots. Side-panel
  ladder still shows the live view.
- **PDF/DOCX/TXT/MD upload** (binary → base64 → /upload_file).
- **Studio**: 5 artifact kinds, **downloads as PDF/Word/Excel/image** — PDF/Word
  carry REAL paginated text + tables (via parseBlocks + jsPDF splitTextToSize), NOT
  screenshots; image is the only rasterized format.
- **Runtime toggle** (Local·5070 / OpenAI) — greys out OpenAI with a clear message
  when no server key.
- **Theme toggle** (light/dark, persists).

### studio.py / app.py — Streamlit NotebookLM UIs (alternative surfaces)
studio.py = 3-column NotebookLM over telco_ran, fully local. app.py = 5 tabs
matching the architecture SVG. Both reuse agentic_rag_lab.py (must sit in repo root).

---

## Bugs fixed this phase (all diagnosed via error-surfacing)

1. **gpt-5.5 rejects `temperature`** — `400: temperature does not support 0, only
   default (1)`. FIX: removed the `temperature=` arg from ALL OpenAI call sites
   (kept it only for Ollama). This caused BOTH the silent qwen-fallback AND the
   "(no answer)" blanks.
2. **`rrf_fuse` not imported** in serve.py → `NameError` in the patterns. FIX:
   `from rrf import rrf_fuse` (line ~45).
3. **Silent OpenAI→qwen fallback** hid the real error. FIX: `_gen_text` now surfaces
   the reason in the mode badge (e.g. `local:qwen2.5:7b (openai failed: …)`).
4. **Empty local response → blank "(no answer)"**. FIX: `_generate` detects an empty
   Ollama response, falls back to the top passage, badge flags `(empty→extractive)`.
   `/diag/local` diagnoses the root cause (usually a model-name mismatch — check
   `ollama list` vs `LOCAL_GEN_MODEL`).
5. **Window scrolled/expanded the whole page** (flexbox trap). FIX: `.main` got
   `min-height:0; overflow:hidden`, body overflow locked. Chat now scrolls INSIDE
   its own pane; header/sidebars fixed.
6. **`No module named 'docx'`** — python-docx not installed. FIX: clear install hint
   in the upload error; user must run
   `pip install python-docx PyMuPDF beautifulsoup4` in the venv.

---

## PENDING (priority order)

1. **[TOP — GRADED] Write the report/paper.** See "THE GRADED DELIVERABLE" above.
   This is the one thing between the project and submission. ~8 days out. The
   product is done; stop polishing and write.
2. **[verify on box] Install deps + restart:** `pip install python-docx PyMuPDF
   beautifulsoup4`, then `uvicorn serve:app --host 0.0.0.0 --port 8000`. Test:
   - `/diag/openai` → should return `ok:true model:gpt-5.5 reply:OK` (temperature
     fix). If not, paste the error.
   - `/diag/local` → diagnoses any blank-local issue (likely model name; check
     `ollama list`).
   - Deep Research pill on a corpus-GAP query (e.g. "3GPP Release 18 RAN energy
     saving", "5G RedCap scheduling", "O-RAN 7-2x fronthaul split") — NOT a handover
     query (those answer locally). Needs internet.
   - PDF/DOCX upload; resizable columns; inline reasoning disclosure.
3. **[recurring gotcha] Confirm `agentic_rag_lab.py` sits in repo root** next to
   serve.py/studio.py (was missing once → FileNotFoundError).
4. **[optional, if time] Rung 8 RAGAS** — gold-free faithfulness/answer-relevancy;
   answers "what about no-gold real-world uploads". Highest-value remaining rung
   AFTER the paper, not before.

---

## Deep Research — how it behaves (for the demo)

- Fires the web lane only when asked (the 🌐 pill). Scoped to a telco allow-list
  (3gpp.org, etsi.org, sharetechnote, techplayon, wikipedia, rfwireless-world, …).
- Fetches → cleans → **ingests into Qdrant permanently** → answers over the enriched
  corpus. Trace shows web search → per-URL fetch (✓/✗ reason) → ingest → answer.
- Degrades cleanly: no internet / no allowed sources → answers from local corpus.
- **Killer demo sequence:** (1) local handover query with Agent → 0 LLM, "corpus is
  enough"; (2) Deep Research on a gap query → fetches + ingests + answers; (3) SAME
  gap query again with plain Agent → now answers locally because Deep Research grew
  the corpus. **The system learned.**
- Local corpus covers ONLY handover/RLF/mobility/beam/reselection (16 scenarios
  TS-01…TS-16). Gap topics that force the web lane: Rel-18 energy saving, RedCap,
  O-RAN splits, NR positioning, NTN/satellite, beam-management detail.

---

## Key locks / design decisions (carry forward)

- The retrieval SYSTEM uses ZERO gold labels; gold is the dev-time ruler only.
- Only ranks transfer across lanes → fuse by RRF, not score. Small pools protect a
  weak judge; confidence gate overrules a collapsed reranker.
- Config-stamping every results file caught model drift twice (load-bearing habit).
- `.env` IS loaded (config.py calls `load_dotenv()`); key confirmed present
  `GENERATOR=openai`, `OPENAI_MODEL=gpt-5.5`.
- Operating rules unchanged: step-wise, honest analysis over validation, terse/
  directive, file-ready outputs, depth over coverage.

---

## Environment (unchanged from HANDOFF.md)

Windows 11, Intel Ultra 9 285K, RTX 5070 (12 GB), 32 GB RAM. Ollama native
(nomic-embed-text 768-D, qwen2.5:7b). Qdrant server mode, `telco_ran`, 26,203
points. Reranker bge-reranker-base. `.venv` at `%USERPROFILE%	elco-rag`.
Bring-up: `python bootstrap.py`. Corpus guard: `python eval\check_corpus.py`.
Serve UI over http: `cd ui; python -m http.server 5173` → http://localhost:5173/console.html
(NOT file://, or CORS blocks the engine).
