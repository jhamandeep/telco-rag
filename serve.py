"""
serve.py - FastAPI sidecar exposing the TelcoRAG agent as an HTTP API.

This is the PRODUCT boundary: the same agent.run() the eval harness measures,
wrapped as a service a frontend can call. Nothing about retrieval changes - the
sidecar adds two things a product needs that an eval harness does not:
  1. answer GENERATION: retrieved chunks -> a grounded answer with [n] citations
     (the NotebookLM core - "answer, sourced from your documents").
  2. a JSON contract carrying the answer, the citation cards (doc text), AND the
     agent's escalation trace (step1->2->3, cost) - so the UI can show HOW the
     answer was found, which is this project's differentiator over a plain RAG.

Endpoints:
  GET  /health            - liveness + corpus size (guards the 16-point wipe)
  POST /ask {query}       - run the agent, generate a grounded answer, return
                            {answer, citations[], trace{steps, cost}}
  GET  /sources           - corpus manifest (which specs/scenarios are loaded)

Run (Qdrant server + Ollama up):
    pip install fastapi uvicorn
    uvicorn serve:app --host 0.0.0.0 --port 8000
    # docs at http://localhost:8000/docs  (Swagger, try /ask live)

Generation uses the SAME local qwen (config.LOCAL_GEN_MODEL) by default, so the
whole product runs offline on the workstation - no external API, sidestepping
the OpenAI-timeout that blocks the web path. REWRITE_BACKEND still switches the
agent's rewrite step independently.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from store import get_client
from embed import embed_one
from bm25_tool import ensure_index
from rrf import rrf_fuse
from rerank import rerank
from rewrite import rewrite_query, active_generator
from agent import Agent, _label

app = FastAPI(title="TelcoRAG", version="1.0")
# CORS open so a locally-served React dev server (any port) can call this.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_client = None
_bm25 = None
_agent = None


def _boot():
    """Lazy singletons: build the agent + index once, on first request."""
    global _client, _bm25, _agent
    if _agent is None:
        _client = get_client()
        _bm25 = ensure_index(_client)
        tools = {"dense": lambda q, k: _client.query_points(
                    config.COLLECTION, query=embed_one(q), limit=k,
                    with_payload=True).points,
                 "bm25": lambda q, k: _bm25.search(q, k=k),
                 "rewrite": rewrite_query, "rerank": rerank}
        _agent = Agent(tools, k=10, lane_k=10, budget=3, ce_floor=0.5)
    return _agent


# ---------------------------------------------------------------------------
# The five agentic patterns (ReAct / multi-hop / reflection / planning / plain)
# from agentic_rag_lab.py, wired to run over the SAME hybrid telco retrieval and
# local qwen brain. Loaded UI-free (the lab has no __main__ guard) and without
# requiring an OpenAI key. Each pattern is a generator yielding trace events.
# ---------------------------------------------------------------------------
_patterns = None
CE_FLOOR = 0.5


def _use_openai():
    """True iff the toggle wants OpenAI AND a key is present."""
    return (os.getenv("GENERATOR", config.GENERATOR) == "openai"
            and bool(os.getenv("OPENAI_API_KEY")))


def _brain(messages, model=None):
    """LLM brain for the patterns: OpenAI iff the runtime toggle selects it AND
    a key is set; otherwise local qwen (falls back to qwen on any OpenAI error)."""
    if _use_openai():
        try:
            from openai import OpenAI
            r = OpenAI().chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages)          # no temperature: some models allow only default
            return r.choices[0].message.content
        except Exception:
            pass
    sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = "\n".join((("User: " if m["role"] == "user" else "Assistant: ") + m["content"])
                      for m in messages if m["role"] != "system")
    prompt = (f"{sys_txt}\n\n{convo}\n\nAssistant:").strip()
    r = requests.post(f"{config.OLLAMA_URL}/api/generate",
                      json={"model": config.LOCAL_GEN_MODEL, "prompt": prompt,
                            "stream": False, "options": {"temperature": 0.1}},
                      timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _hybrid_search_for_patterns(query, k=4, lane_k=10):
    """The retrieval tool the patterns call: full hybrid pipeline (dense+BM25+RRF
    +gated rerank), shaped to the lab's {source, text, score} contract."""
    agent = _boot()
    dense = _client.query_points(config.COLLECTION, query=embed_one(query),
                                 limit=lane_k, with_payload=True).points
    lex = _bm25.search(query, k=lane_k)
    fused = rrf_fuse(dense, lex)
    pool = [h for h, _ in fused[:max(k, 10)]]
    judged = rerank(query, pool)
    ce_max = max((s for _, s in judged), default=0.0)
    ordered = ([h for h, _ in fused[:k]] if ce_max < CE_FLOOR
               else [h for h, _ in judged[:k]])
    sc = {id(h): s for h, s in judged}
    out = []
    for h in ordered:
        p = h.payload or {}
        out.append({"source": _label(p), "text": (p.get("text", "") or ""),
                    "score": round(sc.get(id(h), 0.0), 3), "kind": p.get("source", "?")})
    return out


def _load_patterns():
    global _patterns
    if _patterns is not None:
        return _patterns
    lab_path = Path(__file__).resolve().parent / "agentic_rag_lab.py"
    if not lab_path.exists():
        _patterns = {}
        return _patterns
    src = lab_path.read_text(encoding="utf-8")
    idx = src.find("\n# UI\n")
    logic = src[:idx] if idx != -1 else src
    had = "OPENAI_API_KEY" in os.environ
    if not had:
        os.environ["OPENAI_API_KEY"] = "sk-placeholder-import-only"

    class _Noop:
        def __getattr__(self, k): return self
        def __call__(self, *a, **k): return self
    # the lab does `import streamlit as st` at its top; the pattern functions
    # never call st (only the lab's UI does, which we don't exec). Stub it in
    # sys.modules so the import succeeds even if streamlit isn't installed here.
    _stub_st = None
    if "streamlit" not in sys.modules:
        import types as _types
        _stub_st = _types.ModuleType("streamlit")
        sys.modules["streamlit"] = _stub_st
    ns = {"__name__": "serve_lab_patterns", "__file__": str(lab_path),
          "st": _Noop(), "call": _brain, "search_documents": _hybrid_search_for_patterns}
    try:
        exec(compile(logic, str(lab_path), "exec"), ns)
    finally:
        if not had:
            os.environ.pop("OPENAI_API_KEY", None)
        if _stub_st is not None:
            del sys.modules["streamlit"]         # don't leave the stub around
    for fn in ("pattern_plain_rag", "pattern_react", "pattern_multihop",
               "pattern_reflection", "pattern_planning"):
        if fn in ns:
            ns[fn].__globals__["call"] = _brain
            ns[fn].__globals__["search_documents"] = _hybrid_search_for_patterns
    _patterns = ns.get("PATTERNS", {})
    return _patterns


# ---- answer generation (grounded, cited) ----------------------------------
GEN_SYS = (
    "You are a 5G RAN troubleshooting assistant. Answer the engineer's question "
    "USING ONLY the numbered context passages. Cite every claim with its passage "
    "number, e.g. [1].\n"
    "- If the context only PARTIALLY covers the question, give the grounded partial "
    "answer anyway, then add a short 'Not covered by the sources:' line listing what "
    "is missing. Do NOT refuse outright when partial evidence exists.\n"
    "- Passages whose label contains 'web' are UNVERIFIED web results: you may use "
    "them but flag them explicitly as an 'unverified web source'.\n"
    "- The 3GPP specifications contain NO vendor CLI or troubleshooting commands. If "
    "asked to produce commands, DO NOT invent them: name the specific parameters to "
    "check instead, and state that exact commands are vendor-specific and outside the "
    "specification scope (unless a passage explicitly gives a command).\n"
    "Be concise and technical."
)


def _gen_text(prompt, temperature=0.0, force_openai=False):
    """Single generation entrypoint that respects the Local/OpenAI toggle.
    OpenAI iff selected AND keyed; else local qwen. `force_openai=True` uses the
    frontier model whenever a key is present (used for quality-sensitive artifacts
    like diagrams), regardless of the runtime toggle. On an OpenAI error, falls
    back to qwen BUT surfaces the reason in the mode string so failures are
    visible in the UI (e.g. a bad model name) instead of silently going local."""
    use_oa = _use_openai() or (force_openai and bool(os.getenv("OPENAI_API_KEY")))
    if use_oa:
        try:
            from openai import OpenAI
            mdl = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            r = OpenAI().chat.completions.create(
                model=mdl,
                messages=[{"role": "user", "content": prompt}])  # default temperature
            return r.choices[0].message.content.strip(), f"openai:{mdl}"
        except Exception as e:
            # do NOT hide it: carry the reason into the fallback label
            _gen_text._last_openai_error = f"{type(e).__name__}: {e}"[:200]
            _openai_err = _gen_text._last_openai_error
            try:
                rr = requests.post(f"{config.OLLAMA_URL}/api/generate",
                                   json={"model": config.LOCAL_GEN_MODEL, "prompt": prompt,
                                         "stream": False, "options": {"temperature": temperature}},
                                   timeout=180)
                rr.raise_for_status()
                return rr.json().get("response", "").strip(), \
                    f"local:{config.LOCAL_GEN_MODEL} (openai failed: {_openai_err})"
            except Exception:
                return f"(both backends failed) openai: {_openai_err}", "error"
    r = requests.post(f"{config.OLLAMA_URL}/api/generate",
                      json={"model": config.LOCAL_GEN_MODEL, "prompt": prompt,
                            "stream": False, "options": {"temperature": temperature}},
                      timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip(), f"local:{config.LOCAL_GEN_MODEL}"


_gen_text._last_openai_error = None


def _generate(query, hits, k=5, history=None, force_openai=False):
    """Compose the top hits into a grounded, cited answer (toggle-aware).
    Falls back to an extractive summary if generation is unavailable OR empty.
    `history` makes it conversational; `force_openai` routes to the frontier model
    for synthesis-heavy queries (call flows, comparisons, command requests)."""
    ctx_hits = hits[:k]
    if not ctx_hits:
        return "No sources were retrieved for this query. Try rephrasing.", "no-context"
    context = "\n\n".join(
        f"[{i}] ({_label(h.payload or {})}) {(h.payload or {}).get('text','')[:600]}"
        for i, h in enumerate(ctx_hits, 1))
    convo = ""
    if history:
        convo = ("Conversation so far (for context; cite only the numbered "
                 "sources below):\n" + "\n".join(
                     f"{h.get('role','user')}: {(h.get('content','') or '')[:300]}"
                     for h in history[-4:]) + "\n\n")
    prompt = f"{GEN_SYS}\n\n{convo}Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    try:
        text, mode = _gen_text(prompt, force_openai=force_openai)
        if not text or not text.strip():
            # generation succeeded but produced nothing - show the top passage
            top = ctx_hits[0].payload.get("text", "")
            return (f"The generator returned an empty response. Most relevant "
                    f"passage [1]: {top[:500]}"), f"{mode} (empty→extractive)"
        return text, mode
    except Exception as e:
        top = ctx_hits[0].payload.get("text", "") if ctx_hits else ""
        return (f"(generation unavailable: {type(e).__name__}: {e}) Most relevant "
                f"passage [1]: {top[:400]}"), "fallback"


CONDENSE_SYS = (
    "You rewrite a user's latest message into ONE standalone search query for a 5G "
    "RAN specification + troubleshooting knowledge base. Resolve every pronoun and "
    "back-reference ('it', 'that', 'this', 'the same', 'those metrics') using the "
    "conversation so the query stands on its own. Keep it concise and keyword-rich. "
    "Output ONLY the rewritten query, no quotes, no preamble.")


def _condense(query, history):
    """History-aware retrieval: turn a follow-up into a self-contained query so
    dense/lexical search can actually find the right clause. Returns
    (search_query, condensed_bool). No history -> unchanged (first turn is free)."""
    if not history:
        return query, False
    convo = "\n".join(f"{h.get('role','user')}: {(h.get('content','') or '')[:400]}"
                      for h in history[-6:])
    prompt = (f"{CONDENSE_SYS}\n\nConversation:\n{convo}\n\n"
              f"Latest message: {query}\n\nStandalone query:")
    try:
        out, _ = _gen_text(prompt)
        out = (out or "").strip().splitlines()[0].strip()
        # strip search-instruction wrappers the model sometimes adds, and quotes
        out = re.sub(r'^\s*(search(\s+for)?|find|look\s*up|lookup|retrieve|query)\s*[:\-]?\s*',
                     '', out, flags=re.I).strip()
        out = out.strip('"').strip("'").strip()
        if out and len(out) > 3 and out.lower() != query.lower():
            return out, True
    except Exception:
        pass
    return query, False


_FRONTIER_INTENT = re.compile(
    r"\b(call\s*flow|end.to.end|step[-\s]?by[-\s]?step|walk\s*through|construct|"
    r"design|compare|comparison|architecture|procedures?|sequence|diagrams?|"
    r"explain\s+how|derive|commands?|cli|configure|configuration)\b",
    re.I)


def _wants_frontier(q):
    """Synthesis-heavy or command-style queries route to the frontier model: the
    local 7B refuses or hallucinates on these (see the console evaluation), while a
    frontier model gives structured, better-grounded answers and respects the
    no-command guard. Simple lookups stay local."""
    return bool(_FRONTIER_INTENT.search(q or ""))


# ---- schema ---------------------------------------------------------------
class AskRequest(BaseModel):
    query: str
    generate: bool = True       # UI can request retrieval-only (no LLM answer)
    history: list = []          # recent [{role, content}] turns for conversational memory


class ArtifactRequest(BaseModel):
    query: str
    kind: str = "report"        # report | table | diagram | mindmap | studyguide | quiz
    prefer_frontier: bool = True  # artifacts prefer the frontier model for quality


def _ingest_text(raw_text, source_name, spec_hint=None):
    """Chunk + embed + upsert an uploaded document into the SAME collection,
    tagged source='upload' so it is distinguishable yet fully searchable.
    Reuses chunk_document/embed_batch/PointStruct exactly as ingest_specs does,
    so an uploaded doc becomes a first-class corpus citizen."""
    import uuid as _uuid
    from chunker import chunk_document
    from embed import embed_batch
    from qdrant_client.models import PointStruct
    spec = spec_hint or "upload"
    chunks = chunk_document(raw_text, spec=spec, mode=config.CHUNK_MODE)
    if not chunks:
        return {"chunks": 0, "source": source_name}
    c = get_client()
    pts, EB = [], 32
    for i in range(0, len(chunks), EB):
        grp = chunks[i:i + EB]
        vecs = embed_batch([ch.text for ch in grp])
        for ch, v in zip(grp, vecs):
            pid = str(_uuid.uuid5(_uuid.NAMESPACE_URL,
                                  f"telcorag:upload:{source_name}:{ch.cid}"))
            pl = {**ch.payload(), "source": "upload", "upload_name": source_name}
            pts.append(PointStruct(id=pid, vector=v, payload=pl))
    c.upsert(config.COLLECTION, points=pts)
    global _bm25, _agent
    _bm25 = _agent = None          # force lexical index rebuild on next query
    return {"chunks": len(chunks), "source": source_name, "spec": spec}


ARTIFACT_SYS = {
    "report": ("Write a concise technical incident report for a 5G RAN engineer, "
               "using ONLY the numbered context. Sections: Summary, Likely cause, "
               "Evidence (cite [n]), Recommended checks. Cite every claim."),
    "table": ("Extract the key parameters/timers/causes from the context as a "
              "GitHub-flavoured markdown TABLE with columns Parameter | Meaning | "
              "Typical value or effect | Source. Use ONLY the context; cite [n] "
              "in the Source column. Output only the table."),
    "diagram": ("Produce a VALID Mermaid flowchart inside one ```mermaid ... ``` "
                "block for the fault mechanism or procedure in the context. RULES: "
                "first line must be 'flowchart TD'. Put EVERY node label in DOUBLE "
                "QUOTES, e.g. A[\"Error indication (GTP-U error)\"] -- parentheses, "
                "commas and colons are allowed ONLY inside the quotes. Node ids are "
                "short and alphanumeric (A, B, N1). Edges: A --> B or A -->|\"x\"| B. "
                "Do NOT use ::: class styling or classDef. 8-14 nodes, one edge per "
                "line. Use ONLY the context. Output only the mermaid block."),
    "mindmap": ("Produce a VALID Mermaid mindmap inside one ```mermaid ... ``` block. "
                "RULES: first line exactly 'mindmap'; second line 'root((Topic))'; "
                "then indented branches, TWO spaces per depth level. Keep each branch "
                "text SHORT with no parentheses or special characters. Do NOT use ::: "
                "styling. Branches come from the context ONLY. Output only the block."),
    "studyguide": ("Write a study guide from the context ONLY: 3-5 key concepts each "
                   "with a one-line definition, then 3 'things to remember'. Cite [n]. "
                   "Markdown."),
    "quiz": ("Write 4 multiple-choice questions testing the context ONLY. Each: a "
             "question, options A-D, then 'Answer: X' and a one-line why with [n]. "
             "Markdown."),
}


def _citation(h, rank):
    p = h.payload or {}
    return {"n": rank, "label": _label(p),
            "source": p.get("source", "?"),
            "spec": p.get("spec"), "clause": p.get("clause"),
            "scenario_id": p.get("id"),
            "text": (p.get("text", "") or "")[:600]}


def _step(s):
    return {"n": s.n, "action": s.action, "detail": s.detail,
            "top": s.top_label, "sufficient": s.sufficient}


# ---- endpoints ------------------------------------------------------------
@app.get("/diag/local")
def diag_local():
    """Directly test the local Ollama generator. Confirms the model is pulled
    and responding, and shows the raw reply - to diagnose blank local answers."""
    try:
        # list models to confirm the configured one exists
        tags = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=10).json()
        names = [m.get("name", "") for m in tags.get("models", [])]
        model_present = any(config.LOCAL_GEN_MODEL in n for n in names)
        r = requests.post(f"{config.OLLAMA_URL}/api/generate",
                          json={"model": config.LOCAL_GEN_MODEL,
                                "prompt": "Reply with the single word OK.",
                                "stream": False}, timeout=60)
        r.raise_for_status()
        reply = r.json().get("response", "")
        return {"ok": bool(reply.strip()), "model": config.LOCAL_GEN_MODEL,
                "model_pulled": model_present, "available_models": names,
                "reply": reply.strip()[:120],
                "hint": None if model_present else
                f"'{config.LOCAL_GEN_MODEL}' not in Ollama. Run: ollama pull {config.LOCAL_GEN_MODEL}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300],
                "hint": "Is Ollama running? Start it and check OLLAMA_URL."}


@app.get("/diag/openai")
def diag_openai():
    """Directly test the configured OpenAI model+key and return the exact result
    or error. Use this to diagnose why generation falls back to local."""
    if not os.getenv("OPENAI_API_KEY"):
        return {"ok": False, "reason": "no OPENAI_API_KEY in server env"}
    mdl = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=mdl, messages=[{"role": "user", "content": "reply with the single word OK"}])
        return {"ok": True, "model": mdl, "reply": r.choices[0].message.content.strip()}
    except Exception as e:
        return {"ok": False, "model": mdl, "error": f"{type(e).__name__}: {e}"[:400],
                "hint": "If this is a model-not-found error, OPENAI_MODEL in .env "
                        "is not a model your key can call. Try gpt-4o or gpt-4o-mini."}


# ---------------------------------------------------------------------------
# DEEP WEB RESEARCH - the agent's last-resort escalation: when the local corpus
# is insufficient, fetch telco/3GPP sources from an ALLOW-LIST, ingest them into
# the same Qdrant, then answer grounded in the enriched corpus. Telco-scoped so
# it stays a domain expert, not a general crawler. Reached only on demand.
# ---------------------------------------------------------------------------
TELCO_ALLOW = [
    "3gpp.org", "etsi.org", "sharetechnote.com", "techplayon.com",
    "wikipedia.org", "rfwireless-world.com", "5g-tools.com",
    "commsbrief.com", "telecomtrainer.com",
]


def _web_search_urls(query, k=4):
    """Find candidate telco URLs via DuckDuckGo HTML (no API key). Returns URLs
    whose host is on the telco allow-list only."""
    try:
        from bs4 import BeautifulSoup
        import urllib.parse
        q = urllib.parse.quote(query + " 5G RAN 3GPP")
        r = requests.get(f"https://html.duckduckgo.com/html/?q={q}",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        urls = []
        for a in soup.select("a.result__a"):
            href = a.get("href", "")
            # ddg wraps in a redirect; extract the real target
            m = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg")
            target = m[0] if m else href
            host = urllib.parse.urlparse(target).netloc.lower()
            if any(dom in host for dom in TELCO_ALLOW):
                urls.append(target)
            if len(urls) >= k:
                break
        return urls
    except Exception:
        return []


def _fetch_clean(url):
    """Fetch a page and extract readable text (telco allow-list enforced)."""
    import urllib.parse
    host = urllib.parse.urlparse(url).netloc.lower()
    if not any(dom in host for dom in TELCO_ALLOW):
        return None, "blocked (not on telco allow-list)"
    try:
        from bs4 import BeautifulSoup
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text[:8000], None
    except Exception as e:
        return None, f"{type(e).__name__}"


@app.post("/research")
def research(payload: dict):
    """Deep web research: search telco sources, fetch + ingest the top pages,
    then answer grounded in the enriched corpus. Body: {query}."""
    query = payload.get("query", "")
    events = []
    urls = _web_search_urls(query)
    events.append({"type": "search", "text": f"telco web search: {query}",
                   "found": len(urls)})
    if not urls:
        events.append({"type": "error",
                       "text": "No telco-scoped sources found (or web unreachable). "
                               "Answering from the local corpus only."})
        agent = _boot()
        tr = agent.run("api", query)
        ans, mode = _generate(query, tr.final_hits)
        return {"query": query, "answer": ans, "generation_mode": mode,
                "events": events, "ingested": 0,
                "citations": [_citation(h, i) for i, h in enumerate(tr.final_hits, 1)]}
    ingested = 0
    for url in urls:
        text, err = _fetch_clean(url)
        if err:
            events.append({"type": "fetch", "url": url, "ok": False, "note": err})
            continue
        res = _ingest_text(text, url, spec_hint="web")
        ingested += res.get("chunks", 0)
        events.append({"type": "fetch", "url": url, "ok": True,
                       "chunks": res.get("chunks", 0)})
    events.append({"type": "ingest", "text": f"added {ingested} chunks from "
                   f"{sum(1 for e in events if e.get('ok'))} sources"})
    # now answer over the enriched corpus
    agent = _boot()
    tr = agent.run("api", query)
    ans, mode = _generate(query, tr.final_hits)
    events.append({"type": "answer", "text": ans})
    return {"query": query, "answer": ans, "generation_mode": mode,
            "events": events, "ingested": ingested,
            "sources_fetched": urls,
            "citations": [_citation(h, i) for i, h in enumerate(tr.final_hits, 1)]}


@app.get("/config")
def get_config():
    """Current generator backend (local qwen vs OpenAI) for the UI toggle."""
    return {"generator": "openai" if os.getenv("OPENAI_API_KEY")
            and os.getenv("GENERATOR", config.GENERATOR) == "openai" else "local",
            "openai_available": bool(os.getenv("OPENAI_API_KEY")),
            "local_model": config.LOCAL_GEN_MODEL,
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")}


@app.post("/config")
def set_config(payload: dict):
    """Switch the generator backend live. Sets GENERATOR in the process env so
    subsequent /ask, /pattern, /artifact calls use it. 'openai' requires
    OPENAI_API_KEY to already be set; otherwise stays local and says so."""
    want = payload.get("generator", "local").lower()
    if want == "openai" and not os.getenv("OPENAI_API_KEY"):
        return {"ok": False, "generator": "local",
                "error": "OPENAI_API_KEY not set on the server - staying local. "
                         "Set the key in the environment to enable OpenAI."}
    os.environ["GENERATOR"] = want
    config.GENERATOR = want          # keep config in sync for code that reads it
    global _patterns
    _patterns = None                 # patterns re-read the brain on next load
    return {"ok": True, "generator": want,
            "model": (os.getenv("OPENAI_MODEL", "gpt-4o-mini") if want == "openai"
                      else config.LOCAL_GEN_MODEL)}


@app.get("/health")
def health():
    try:
        c = get_client()
        n = c.count(config.COLLECTION, exact=True).count
        ok = n > 1000
        return {"status": "ok" if ok else "degraded",
                "collection": config.COLLECTION, "points": n,
                "qdrant_mode": config.QDRANT_MODE,
                "warning": None if ok else
                "corpus far below ~26,203 - spec lane may be absent (wipe?)"}
    except Exception as e:
        return {"status": "down", "error": f"{type(e).__name__}: {e}"}


@app.post("/ask")
def ask(req: AskRequest):
    agent = _boot()
    search_q, condensed = _condense(req.query, req.history)   # history-aware retrieval
    tr = agent.run("api", search_q)
    use_frontier = _wants_frontier(req.query)                 # auto-route hard synthesis
    answer, gen_mode = ("", "skipped")
    if req.generate:
        answer, gen_mode = _generate(req.query, tr.final_hits, history=req.history,
                                     force_openai=use_frontier)
    return {
        "query": req.query,
        "search_query": search_q,        # what retrieval actually ran on
        "condensed": condensed,          # True if the follow-up was rewritten
        "routed_frontier": use_frontier, # True if forced to the frontier model
        "answer": answer,
        "generation_mode": gen_mode,
        "citations": [_citation(h, i)
                      for i, h in enumerate(tr.final_hits, 1)],
        "trace": {
            "steps": [_step(s) for s in tr.steps],
            "stop_reason": tr.stop_reason,
            "tool_calls": tr.tool_calls,
            "llm_calls": tr.llm_calls,
            "stop_step": tr.steps[-1].n if tr.steps else 0,
        },
        "agent_generator": active_generator(),
    }


@app.post("/upload_file")
def upload_file(payload: dict):
    """Ingest an uploaded PDF or DOCX. Body: {name, b64} where b64 is the base64
    file content. Extracts text server-side (PyMuPDF for PDF, python-docx for
    DOCX), then chunks + embeds into the corpus. The NotebookLM upload core."""
    import base64
    name = payload.get("name", "untitled")
    b64 = payload.get("b64", "")
    if not b64:
        return {"ok": False, "error": "no file content"}
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return {"ok": False, "error": "invalid base64"}
    text = ""
    low = name.lower()
    try:
        if low.endswith(".pdf"):
            try:
                import fitz
            except ModuleNotFoundError:
                return {"ok": False, "error": "PDF support needs PyMuPDF. Install: "
                        "pip install PyMuPDF"}
            doc = fitz.open(stream=raw, filetype="pdf")
            text = "\n".join(p.get_text() for p in doc)
            doc.close()
        elif low.endswith(".docx"):
            try:
                import docx
            except ModuleNotFoundError:
                return {"ok": False, "error": "DOCX support needs python-docx. Install: "
                        "pip install python-docx"}
            import io
            d = docx.Document(io.BytesIO(raw))
            text = "\n".join(p.text for p in d.paragraphs)
        elif low.endswith((".txt", ".md")):
            text = raw.decode("utf-8", errors="ignore")
        else:
            return {"ok": False, "error": f"unsupported type: {name}"}
    except Exception as e:
        return {"ok": False, "error": f"extraction failed: {type(e).__name__}: {e}"}
    if not text.strip():
        return {"ok": False, "error": "no extractable text (scanned PDF?)"}
    res = _ingest_text(text, name, payload.get("spec"))
    return {"ok": True, "chars": len(text), **res}


@app.post("/upload")
def upload(payload: dict):
    """Ingest an uploaded document (raw text already extracted client-side, or
    plain text). Body: {name, text, spec?}. Grows the corpus live - the
    NotebookLM 'bring your own docs' core."""
    name = payload.get("name", "untitled")
    text = payload.get("text", "")
    if not text.strip():
        return {"ok": False, "error": "empty document text"}
    res = _ingest_text(text, name, payload.get("spec"))
    return {"ok": True, **res}


@app.post("/artifact")
def artifact(req: ArtifactRequest):
    """Retrieve for the query, then have the generator emit a structured artifact
    grounded in the retrieved context. Prefers the frontier model (quality matters
    for diagrams/mindmaps); falls back to local qwen if no key is present."""
    agent = _boot()
    tr = agent.run("api", req.query)
    sys_p = ARTIFACT_SYS.get(req.kind, ARTIFACT_SYS["report"])
    ctx = "\n\n".join(
        f"[{i}] ({_label(h.payload or {})}) {(h.payload or {}).get('text','')[:600]}"
        for i, h in enumerate(tr.final_hits[:6], 1))
    prompt = f"{sys_p}\n\nContext:\n{ctx}\n\nTopic: {req.query}\n\nOutput:"
    try:
        out, mode = _gen_text(prompt, force_openai=req.prefer_frontier)
    except Exception as e:
        out, mode = f"(generation unavailable: {type(e).__name__})", "error"
    return {"kind": req.kind, "content": out, "mode": mode,
            "citations": [_citation(h, i) for i, h in enumerate(tr.final_hits[:6], 1)],
            "trace": {"tool_calls": tr.tool_calls, "llm_calls": tr.llm_calls,
                      "stop_step": tr.steps[-1].n if tr.steps else 0}}


@app.get("/patterns")
def list_patterns():
    """Names of the available agentic patterns (for the console selector)."""
    pats = _load_patterns()
    return {"patterns": list(pats.keys()),
            "available": bool(pats),
            "note": None if pats else
            "agentic_rag_lab.py not found next to serve.py - only the agent is available"}


@app.post("/pattern")
def run_pattern(payload: dict):
    """Run one of the five agentic patterns over the hybrid telco retrieval and
    return its full event trace (think/search/chunks/draft/critic/plan/answer)
    plus the final answer and citations. This exposes the pattern COMPARISON
    that is the heart of the project - not just the single rung-7d agent."""
    pats = _load_patterns()
    name = payload.get("pattern", "")
    query = payload.get("query", "")
    if name not in pats:
        return {"error": f"unknown pattern '{name}'",
                "available": list(pats.keys())}
    events, answer, sources = [], "", []
    tool_calls = 0
    try:
        for ev in pats[name](query, None):
            t = ev.get("type")
            if t == "chunks":
                tool_calls += 1
                events.append({"type": "chunks",
                               "chunks": [{"source": c["source"],
                                           "score": c.get("score"),
                                           "kind": c.get("kind", "?"),
                                           "text": (c.get("text", "") or "")[:200]}
                                          for c in ev["chunks"]]})
            elif t == "answer":
                answer = ev.get("text", "")
                sources = ev.get("sources", [])
                events.append({"type": "answer", "text": answer,
                               "hops": ev.get("hops")})
            elif t == "plan":
                events.append({"type": "plan", "subqs": ev.get("subqs", [])})
            elif t == "critic":
                events.append({"type": "critic", "grounded": ev.get("grounded"),
                               "text": ev.get("text", "")})
            else:  # think / search / subq / draft
                events.append({"type": t, "text": ev.get("text", ""),
                               "hop": ev.get("hop")})
    except Exception as e:
        events.append({"type": "error", "text": f"{type(e).__name__}: {e}"})
    return {"pattern": name, "query": query, "answer": answer,
            "sources": sources, "events": events,
            "tool_calls": tool_calls, "brain": _brain_label()}


def _brain_label():
    return (f"openai:{os.getenv('OPENAI_MODEL','gpt-4o-mini')}"
            if _use_openai() else f"local:{config.LOCAL_GEN_MODEL}")


@app.get("/sources")
def sources():
    """Corpus manifest: distinct specs + scenario count, for the UI's panel."""
    c = get_client()
    specs, scen, offset = {}, 0, None
    while True:
        pts, offset = c.scroll(config.COLLECTION, limit=1024,
                               with_payload=True, with_vectors=False,
                               offset=offset)
        for pt in pts:
            p = pt.payload or {}
            if p.get("source") == "troubleshooting" or p.get("id", "").startswith("TS-"):
                scen += 1
            else:
                specs[p.get("spec", "?")] = specs.get(p.get("spec", "?"), 0) + 1
        if offset is None:
            break
    return {"specs": specs, "scenario_chunks": scen,
            "total": sum(specs.values()) + scen}


@app.get("/uploads")
def uploads():
    """List uploaded files (distinct upload_name) with chunk counts, for the UI's
    file manager. Only points tagged source='upload' are user files."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    c = get_client()
    flt = Filter(must=[FieldCondition(key="source",
                                      match=MatchValue(value="upload"))])
    files, offset = {}, None
    while True:
        pts, offset = c.scroll(config.COLLECTION, limit=1024, with_payload=True,
                               with_vectors=False, offset=offset, scroll_filter=flt)
        for pt in pts:
            nm = (pt.payload or {}).get("upload_name", "(unnamed)")
            files[nm] = files.get(nm, 0) + 1
        if offset is None:
            break
    return {"files": [{"name": k, "chunks": v} for k, v in sorted(files.items())],
            "count": len(files)}


@app.post("/delete_file")
def delete_file(payload: dict):
    """Delete every point of one uploaded file by name, then force the lexical
    index + agent to rebuild so the removed doc is gone from retrieval too."""
    name = (payload or {}).get("name", "").strip()
    if not name:
        return {"ok": False, "error": "name required"}
    from qdrant_client.models import (Filter, FieldCondition, MatchValue,
                                      FilterSelector)
    c = get_client()
    flt = Filter(must=[
        FieldCondition(key="source", match=MatchValue(value="upload")),
        FieldCondition(key="upload_name", match=MatchValue(value=name))])
    try:
        n = c.count(config.COLLECTION, exact=True, count_filter=flt).count
        c.delete(config.COLLECTION, points_selector=FilterSelector(filter=flt))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    global _bm25, _agent
    _bm25 = _agent = None
    return {"ok": True, "deleted_chunks": n, "name": name}


@app.post("/reload")
def reload_indexes():
    """Drop the cached BM25 index and agent so the next query rebuilds them from the
    CURRENT Qdrant corpus. Call this after ingesting new scenarios/specs from a
    separate process (e.g. ingest_troubleshooting.py) instead of restarting serve.py:
    dense retrieval sees new points immediately, but the in-memory BM25 lane does not
    until it is rebuilt."""
    global _bm25, _agent
    _bm25 = _agent = None
    try:
        n = get_client().count(config.COLLECTION, exact=True).count
    except Exception:
        n = None
    return {"ok": True, "reloaded": True, "corpus_points": n}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
