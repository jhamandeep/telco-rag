"""
studio.py - TelcoRAG Studio: the five-pattern agentic lab extended into a
NotebookLM-style workspace, running FULLY LOCAL on one machine (RTX 5070 +
qwen2.5:7b via Ollama), pointed at the proven telco_ran corpus.

What this is and isn't
----------------------
It reuses the five agentic patterns from agentic_rag_lab.py UNCHANGED (Plain RAG,
ReAct, Multi-hop, Reflection, Planning) - each a generator that yields live trace
events. The only swap is the retrieval TOOL underneath them: instead of the lab's
dense-only search over learner_corpus, every pattern now calls the TelcoRAG hybrid
retriever (dense + BM25 + RRF + gated rerank) over telco_ran - the pipeline this
project measured to MRR 0.936.

On top of that it adds a NotebookLM-style STUDIO column: grounded artifacts
generated from retrieved context - Report, Table, Mind map (Mermaid), Study guide,
Quiz. Only artifacts that are honestly achievable locally; no fake audio/video.

Single-machine design (RTX 5070 + Intel Ultra 9, no external dependency)
------------------------------------------------------------------------
  embeddings   nomic-embed-text   Ollama  (GPU)
  retrieval    dense+BM25+RRF     local Qdrant + rank_bm25
  rerank       bge-reranker-base  GPU
  reasoning    qwen2.5:7b         Ollama  (GPU)   <- the LLM brain, LOCAL
  generation   qwen2.5:7b         Ollama  (GPU)
  UI           Streamlit          localhost

OpenAI is OPTIONAL: if OPENAI_API_KEY is set AND reachable, the brain uses it for
sharper reasoning; otherwise everything falls back to local qwen so the whole
app runs with the network off - which is what you want for a live demo.

Run:
    pip install streamlit
    # Qdrant (26k corpus) + Ollama must be up, as for the eval harness
    streamlit run studio.py
"""
import os
import sys
import json
import re
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))

import config
from store import get_client
from embed import embed_one
from bm25_tool import ensure_index
from rrf import rrf_fuse
from rerank import rerank
from rewrite import rewrite_query

# ---------------------------------------------------------------------------
# LLM brain: local qwen by default, OpenAI only if configured AND reachable.
# The lab's patterns import `call` and `search_documents` from here.
# ---------------------------------------------------------------------------
_openai_ok = None      # tri-state: None=untested, True/False=probed


def _try_openai():
    global _openai_ok
    if _openai_ok is not None:
        return _openai_ok
    if not os.getenv("OPENAI_API_KEY"):
        _openai_ok = False
        return False
    try:
        from openai import OpenAI
        OpenAI().models.list()           # cheap reachability probe
        _openai_ok = True
    except Exception:
        _openai_ok = False
    return _openai_ok


def _call_local(messages, temperature=0.1):
    """Chat via Ollama qwen. Flatten messages into a single prompt (qwen /api/
    generate is prompt-based); system content leads."""
    sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = "\n".join(
        (("User: " if m["role"] == "user" else "Assistant: ") + m["content"])
        for m in messages if m["role"] != "system")
    prompt = (f"{sys_txt}\n\n{convo}\n\nAssistant:").strip()
    r = requests.post(f"{config.OLLAMA_URL}/api/generate",
                      json={"model": config.LOCAL_GEN_MODEL, "prompt": prompt,
                            "stream": False, "options": {"temperature": temperature}},
                      timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def call(messages, model=None):
    """The brain. Prefers OpenAI iff configured+reachable, else local qwen.
    `model` is accepted for lab compatibility but local runs use LOCAL_GEN_MODEL."""
    if _try_openai():
        try:
            from openai import OpenAI
            resp = OpenAI().chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages, temperature=0.1)
            return resp.choices[0].message.content
        except Exception:
            pass                                    # fall through to local
    return _call_local(messages)


def brain_label():
    return (f"openai:{os.getenv('OPENAI_MODEL','gpt-4o-mini')}"
            if _try_openai() else f"local:{config.LOCAL_GEN_MODEL}")


# ---------------------------------------------------------------------------
# Retrieval TOOL: the proven TelcoRAG hybrid pipeline, shaped to the lab's
# {source, text, score} contract so the five patterns consume it unchanged.
# dense + BM25 -> RRF fuse -> gated cross-encoder rerank.
# ---------------------------------------------------------------------------
_client = None
_bm25 = None
CE_FLOOR = 0.5


def _reset_bm25():
    """Invalidate the cached lexical index so the next query rebuilds it after
    an upload changed the corpus."""
    global _bm25
    _bm25 = None


def _boot_retrieval():
    global _client, _bm25
    if _client is None:
        _client = get_client()
        _bm25 = ensure_index(_client)
    return _client, _bm25


def _label(p):
    if p.get("id"):
        return p["id"]
    return f"{p.get('spec','?')} {p.get('clause','')}".strip()


def search_documents(query, k=4, lane_k=10):
    """Hybrid retrieval over telco_ran: the lab's retrieval tool, upgraded.
    Returns the lab's chunk dicts {source, text, score}. Score here is the
    cross-encoder relevance (0-1) when the judge is confident, else the RRF
    score - so the trace shows a meaningful number either way."""
    client, bm25 = _boot_retrieval()
    dense = client.query_points(config.COLLECTION, query=embed_one(query),
                                limit=lane_k, with_payload=True).points
    lex = bm25.search(query, k=lane_k)
    fused = rrf_fuse(dense, lex)
    pool = [h for h, _ in fused[:max(k, 10)]]
    judged = rerank(query, pool)
    ce_max = max((s for _, s in judged), default=0.0)
    ordered = ([h for h, _ in fused[:k]] if ce_max < CE_FLOOR
               else [h for h, _ in judged[:k]])
    score_of = {id(h): s for h, s in judged}
    out = []
    for h in ordered:
        p = h.payload or {}
        out.append({"source": _label(p),
                    "text": (p.get("text", "") or ""),
                    "score": round(score_of.get(id(h), 0.0), 3),
                    "kind": p.get("source", "?")})
    return out


# import the five patterns AFTER defining call + search_documents, then rebind
# their tool + brain to THIS module's (so they use telco_ran, local qwen).
#
# The lab file (agentic_rag_lab.py) has no __main__ guard - importing it would
# run the lab's OWN Streamlit UI and construct OpenAI() at import. We want only
# its pattern FUNCTIONS. So we exec the lab's source up to its UI marker
# ("# UI") in a namespace we control, with call/search_documents already bound
# to ours. Nothing of the lab's UI or its OpenAI client is executed.
class _LabNS:
    """Holds the lab's pattern functions after a UI-free exec."""
    pass


def _load_lab_patterns():
    lab_path = Path(__file__).resolve().parent / "agentic_rag_lab.py"
    if not lab_path.exists():
        # The lab file must sit next to studio.py (studio reuses its 5 patterns).
        # Rather than crash the whole app, fall back to a single built-in ReAct-
        # style pattern so the UI still runs, and surface a clear instruction.
        st.warning(
            "agentic_rag_lab.py was not found next to studio.py. Copy it into "
            f"the project folder ({lab_path.parent}) to enable all five patterns. "
            "Running with a single built-in retrieval pattern for now.")
        return _builtin_fallback_patterns()
    src = lab_path.read_text(encoding="utf-8")
    # cut at the UI section banner (a line that is exactly "# UI")
    marker = "\n# UI\n"
    idx = src.find(marker)
    logic_src = src[:idx] if idx != -1 else src
    # The lab does `from openai import OpenAI; client_oai = OpenAI()` at module
    # top - that construction raises without a key. Provide a placeholder key
    # JUST for this exec so it succeeds; the lab's client is never used because
    # we rebind the patterns' call() to ours below. Restore the env after.
    had_key = "OPENAI_API_KEY" in os.environ
    if not had_key:
        os.environ["OPENAI_API_KEY"] = "sk-placeholder-lab-import-only"

    class _Noop:
        def __getattr__(self, k): return self
        def __call__(self, *a, **k): return self
    ns = {"__name__": "telcorag_lab_patterns", "__file__": str(lab_path),
          "st": _Noop(), "call": call, "search_documents": search_documents}
    try:
        exec(compile(logic_src, str(lab_path), "exec"), ns)
    except Exception as e:
        if not had_key:
            os.environ.pop("OPENAI_API_KEY", None)
        st.warning(f"Could not load the lab patterns ({type(e).__name__}); "
                   "running with a single built-in pattern.")
        return _builtin_fallback_patterns()
    finally:
        if not had_key:
            os.environ.pop("OPENAI_API_KEY", None)   # keep the box local-only
    ns["call"] = call
    ns["search_documents"] = search_documents
    return ns


def _builtin_fallback_patterns():
    """A minimal self-contained pattern set so studio runs even without the lab
    file. One grounded single-shot retrieval + answer, wired to our hybrid tool."""
    def plain(question, model=None):
        yield {"type": "think", "text": "Single hybrid retrieval, then a grounded answer."}
        chunks = search_documents(question)
        yield {"type": "search", "text": question}
        yield {"type": "chunks", "chunks": chunks}
        ctx = "\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
        ans = call([{"role": "system", "content": "Answer using ONLY the context; cite sources."},
                    {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}])
        yield {"type": "answer", "text": ans,
               "sources": sorted({c["source"] for c in chunks})}
    name = "Hybrid RAG (built-in)"
    return {"PATTERNS": {name: plain},
            "PATTERN_HELP": {name: "Single hybrid retrieval + grounded answer. "
                             "Copy agentic_rag_lab.py into the folder for all five patterns."},
            "pattern_plain_rag": plain, "pattern_react": plain,
            "pattern_multihop": plain, "pattern_reflection": plain,
            "pattern_planning": plain}


_lab = _load_lab_patterns()
# rebind each pattern's globals so calls to call()/search_documents() resolve to
# ours (patterns are module-level functions; their __globals__ is the exec ns).
for _fn in ("pattern_plain_rag", "pattern_react", "pattern_multihop",
            "pattern_reflection", "pattern_planning"):
    _lab[_fn].__globals__["call"] = call
    _lab[_fn].__globals__["search_documents"] = search_documents
PATTERNS = _lab["PATTERNS"]
PATTERN_HELP = _lab["PATTERN_HELP"]


# ---------------------------------------------------------------------------
# STUDIO generators - grounded artifacts from retrieved context (local qwen).
# Each retrieves once for the topic, then composes. Citations are the chunks.
# ---------------------------------------------------------------------------
STUDIO = {
    "Report": ("Write a concise 5G RAN incident report for an engineer using ONLY "
               "the numbered context. Sections: Summary, Likely cause, Evidence "
               "(cite [n]), Recommended checks. Cite every claim with [n]."),
    "Table": ("Extract key parameters, timers, and causes from the context as a "
              "GitHub-flavoured markdown table: columns Parameter | Meaning | "
              "Typical value or effect | Source[n]. Use ONLY the context. Output "
              "only the table."),
    "Mind map": ("Produce a Mermaid mindmap (```mermaid\\nmindmap\\n ...```) of the "
                 "concept in the context. Root = the topic; branches = sub-topics "
                 "and mechanisms from the context ONLY. Output only the mermaid block."),
    "Study guide": ("Write a study guide from the context ONLY: 3-5 key concepts each "
                    "with a one-line definition, then 3 'things to remember'. Cite [n]. "
                    "Markdown."),
    "Quiz": ("Write 4 multiple-choice questions testing the context ONLY. Each: a "
             "question, options A-D, then 'Answer: X' and a one-line why with [n]. "
             "Markdown."),
}


def studio_generate(kind, topic):
    chunks = search_documents(topic, k=6)
    ctx = "\n\n".join(f"[{i}] ({c['source']}) {c['text'][:600]}"
                      for i, c in enumerate(chunks, 1))
    out = call([{"role": "system", "content": STUDIO[kind]},
                {"role": "user", "content": f"Context:\n{ctx}\n\nTopic: {topic}\n\nOutput:"}])
    return out, chunks


# ===========================================================================
# UI - NotebookLM three-zone layout: Sources | Chat+Trace | Studio
# ===========================================================================
st.set_page_config(page_title="TelcoRAG Studio", page_icon="◢", layout="wide")
st.markdown("""
<style>
  .stApp{background:#0b0f14}
  h1,h2,h3,h4,p,span,div,label,li{color:#dfe8f0}
  .sig{font-family:ui-monospace,monospace;color:#3fd1b0;letter-spacing:1.5px;
       font-size:10.5px;text-transform:uppercase}
  .ev{padding:7px 12px;margin:6px 0;border-radius:6px;font-size:13px}
  .e-think{border-left:3px solid #f0b429;background:rgba(240,180,41,.06);font-style:italic;color:#f0b429}
  .e-search{border-left:3px solid #4aa8ff;background:rgba(74,168,255,.06)}
  .e-search .q{font-family:ui-monospace,monospace;color:#4aa8ff;font-weight:600}
  .e-chunk{font-size:11px;color:#8296a8;margin:2px 0 2px 12px;border-left:2px solid #2a3947;padding-left:8px;font-family:ui-monospace,monospace}
  .e-draft{border-left:3px solid #8296a8;background:rgba(130,150,168,.06)}
  .e-ok{border-left:3px solid #3fd1b0;background:rgba(63,209,176,.08);color:#3fd1b0;font-weight:500}
  .e-bad{border-left:3px solid #f0b429;background:rgba(240,180,41,.08);color:#f0b429;font-weight:500}
  .e-final{border-left:3px solid #3fd1b0;background:rgba(63,209,176,.10);padding:11px 14px;margin-top:10px}
  .e-plan{border-left:3px solid #7F77DD;background:rgba(127,119,221,.06)}
  .cite{border:1px solid #1e2a36;border-radius:7px;padding:8px 11px;background:#111820;margin-bottom:6px;font-size:12px}
  .badge{font-family:ui-monospace,monospace;font-size:9px;padding:1px 6px;border-radius:3px}
  .b-ts{background:#0e1f1b;color:#3fd1b0} .b-spec{background:#111c28;color:#4aa8ff} .b-up{background:#241a0e;color:#f0b429}
</style>
""", unsafe_allow_html=True)

# session store for the current answer's context (studio reuses it)
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

st.markdown('<div class="sig">◢◤ TelcoRAG · 5G RAN · single-node RTX 5070 ◢◤</div>',
            unsafe_allow_html=True)

col_src, col_chat, col_studio = st.columns([1, 2.2, 1.3], gap="medium")

# ---- LEFT: sources + corpus + engine ----
with col_src:
    st.markdown('<div class="sig">Sources</div>', unsafe_allow_html=True)
    try:
        c, _ = _boot_retrieval()
        n = c.count(config.COLLECTION, exact=True).count
        if n > 1000:
            st.success(f"{n:,} points · {config.COLLECTION}")
        else:
            st.warning(f"{n} points — corpus looks wiped; run bootstrap.")
    except Exception as e:
        st.error(f"Qdrant/Ollama down: {type(e).__name__}")
    st.caption(f"brain: `{brain_label()}`  ·  embed: `{config.EMBED_MODEL}`")

    st.markdown('<div class="sig" style="margin-top:14px">Add a document</div>',
                unsafe_allow_html=True)
    up = st.file_uploader("PDF / text", type=["pdf", "txt", "md"], label_visibility="collapsed")
    if up and st.button("Ingest", use_container_width=True):
        if up.type == "application/pdf":
            try:
                import fitz
                doc = fitz.open(stream=up.read(), filetype="pdf")
                text = "\n".join(p.get_text() for p in doc); doc.close()
            except Exception as e:
                text = ""; st.error(f"PDF read failed: {e}")
        else:
            text = up.read().decode("utf-8", errors="ignore")
        if text.strip():
            from chunker import chunk_document
            from embed import embed_batch
            from qdrant_client.models import PointStruct
            import uuid
            chunks = chunk_document(text, spec=up.name.split('.')[0][:12],
                                    mode=config.CHUNK_MODE)
            cl, _ = _boot_retrieval()
            pts = []
            for i in range(0, len(chunks), 32):
                grp = chunks[i:i+32]
                vs = embed_batch([ch.text for ch in grp])
                for ch, v in zip(grp, vs):
                    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"studio:{up.name}:{ch.cid}"))
                    pts.append(PointStruct(id=pid, vector=v,
                               payload={**ch.payload(), "source": "upload",
                                        "upload_name": up.name}))
            cl.upsert(config.COLLECTION, points=pts)
            _reset_bm25()                       # rebuild lexical index next query
            st.success(f"Added {len(chunks)} chunks from {up.name}.")

# ---- CENTRE: pattern picker + ask + live trace ----
with col_chat:
    st.markdown('<div class="sig">Agentic retrieval</div>', unsafe_allow_html=True)
    pat = st.selectbox("Pattern", list(PATTERNS.keys()), label_visibility="collapsed")
    st.caption(PATTERN_HELP[pat])
    q = st.text_input("Ask", placeholder="Calls from cars on the expressway die where one sector ends and the next begins…",
                      label_visibility="collapsed")
    go = st.button("Run", type="primary")

    if go and q.strip():
        st.markdown(f'<div class="sig">Trace — {pat.split(" (")[0]} · {brain_label()}</div>',
                    unsafe_allow_html=True)
        captured = []
        for ev in PATTERNS[pat](q, None):
            t = ev["type"]
            if t == "think":
                st.markdown(f'<div class="ev e-think">think · {ev["text"]}</div>', unsafe_allow_html=True)
            elif t == "plan":
                items = "".join(f"<li>{s}</li>" for s in ev["subqs"])
                st.markdown(f'<div class="ev e-plan"><b>Plan</b><ul>{items}</ul></div>', unsafe_allow_html=True)
            elif t == "subq":
                st.markdown(f'<div class="ev e-search">sub-q · <span class="q">{ev["text"]}</span></div>', unsafe_allow_html=True)
            elif t == "search":
                hop = f' · hop {ev["hop"]}' if "hop" in ev else ""
                st.markdown(f'<div class="ev e-search">search{hop} · <span class="q">{ev["text"]}</span></div>', unsafe_allow_html=True)
            elif t == "chunks":
                captured = ev["chunks"]
                html = "".join(f'<div class="e-chunk">[{c["source"]} · {c["score"]}] {c["text"][:120]}…</div>'
                               for c in ev["chunks"])
                st.markdown(html, unsafe_allow_html=True)
            elif t == "draft":
                st.markdown(f'<div class="ev e-draft">draft · {ev["text"][:400]}</div>', unsafe_allow_html=True)
            elif t == "critic":
                cls = "e-ok" if ev["grounded"] else "e-bad"
                lab_ = "GROUNDED" if ev["grounded"] else "NOT GROUNDED"
                st.markdown(f'<div class="ev {cls}">critic · {lab_}<br><span style="font-weight:400">{ev["text"]}</span></div>', unsafe_allow_html=True)
            elif t == "answer":
                extra = f' · {ev["hops"]} hop(s)' if "hops" in ev else ""
                st.markdown(f'<div class="ev e-final"><b>Answer</b>{extra}<br><br>{ev["text"]}</div>', unsafe_allow_html=True)
        st.session_state.last_chunks = captured
        st.caption("Tip: run Plain RAG then an agentic pattern on the same question and compare traces.")

# ---- RIGHT: the NotebookLM studio ----
with col_studio:
    st.markdown('<div class="sig">Studio</div>', unsafe_allow_html=True)
    st.caption("Grounded artifacts from the corpus. Generated locally on the 5070.")
    topic = st.text_input("Topic", value="too-late handover during fast mobility",
                          label_visibility="collapsed", key="studio_topic")
    kind = st.radio("Artifact", list(STUDIO.keys()), horizontal=False, label_visibility="collapsed")
    if st.button(f"Generate {kind}", use_container_width=True, type="primary"):
        with st.spinner(f"Retrieving + composing {kind} on the 5070…"):
            content, chunks = studio_generate(kind, topic)
        if kind == "Mind map":
            body = content.replace("```mermaid", "").replace("```", "").strip()
            st.markdown(f"```mermaid\n{body}\n```")
            with st.expander("source"):
                st.code(body)
        else:
            st.markdown(content)
        st.markdown('<div class="sig" style="margin-top:10px">Grounded in</div>', unsafe_allow_html=True)
        for c in chunks[:4]:
            k = c.get("kind", "?")
            b = "b-up" if k == "upload" else ("b-spec" if k == "spec" else "b-ts")
            kt = {"upload": "yours", "spec": "3GPP"}.get(k, "scenario")
            st.markdown(f'<div class="cite"><span class="badge {b}">{kt}</span> '
                        f'<b>{c["source"]}</b><br>'
                        f'<span style="color:#8296a8">{c["text"][:150]}…</span></div>',
                        unsafe_allow_html=True)
