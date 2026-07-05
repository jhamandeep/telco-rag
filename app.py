"""
app.py - TelcoRAG Streamlit UI (matches notebooklm-architecture.svg).

The SVG: a Streamlit UI on top; uploads flow into ingest + Qdrant; questions hit
retrieval; retrieved context feeds a local generator (grounded Q&A) and an API
sidecar for report/table/diagram. This app IS that top box, talking to serve.py.

Tabs mirror the SVG's UI verbs exactly: Upload · Ask · Report · Table · Diagram.
The differentiator beyond a plain NotebookLM: the "Ask" tab shows the agent's
DIAGNOSTIC LADDER (dense -> bm25 -> rewrite) and its per-query LLM-call cost, so
the viewer sees the adaptive retrieval this project measured (MRR 0.936 at 0.14
LLM calls/query vs a static 1.0/query).

Run (with serve.py already up on :8000, and Qdrant + Ollama behind it):
    pip install streamlit requests
    streamlit run app.py

Point at a different sidecar with:  TELCORAG_API=http://host:8000 streamlit run app.py
"""
import os
import requests
import streamlit as st

API = os.getenv("TELCORAG_API", "http://localhost:8000")
st.set_page_config(page_title="TelcoRAG", page_icon="◢", layout="wide")

# ---- lightweight instrument-panel styling (matches the HTML build's palette) --
st.markdown("""
<style>
  .stApp { background:#0b0f14; }
  h1,h2,h3,h4,p,span,div,label { color:#dfe8f0; }
  .sig { font-family:ui-monospace,monospace; color:#3fd1b0; letter-spacing:1px;
         font-size:11px; text-transform:uppercase; }
  .rung { border-left:2px solid #2a3947; padding:6px 0 6px 14px; margin-left:6px;
          font-family:ui-monospace,monospace; font-size:12px; color:#8296a8; }
  .rung.on { border-color:#3fd1b0; color:#dfe8f0; }
  .rung.stop { border-color:#3fd1b0; }
  .badge { font-family:ui-monospace,monospace; font-size:10px; padding:1px 7px;
           border-radius:4px; }
  .ts { background:#0e1f1b; color:#3fd1b0; }
  .spec { background:#111c28; color:#4aa8ff; }
  .cost { font-family:ui-monospace,monospace; }
  .cite { border:1px solid #1e2a36; border-radius:8px; padding:9px 12px;
          background:#111820; margin-bottom:7px; }
</style>
""", unsafe_allow_html=True)


def api_get(path):
    try:
        return requests.get(API + path, timeout=10).json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def api_post(path, body, timeout=200):
    try:
        return requests.post(API + path, json=body, timeout=timeout).json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def render_ladder(trace):
    """The signature element: the agent's escalation ladder + cost."""
    labels = [("step 1", "dense_search", "semantic · cheap"),
              ("step 2", "+ bm25 · RRF", "exact terms · cheap"),
              ("step 3", "rewrite + refetch", "language model · costly")]
    reached = trace.get("stop_step", 0)
    st.markdown('<div class="sig">Diagnostic ladder</div>', unsafe_allow_html=True)
    for i, (k, act, hint) in enumerate(labels, 1):
        on = i <= reached
        stop = (i == reached)
        cls = "rung" + (" on" if on else "") + (" stop" if stop else "")
        mark = "✓ stopped here" if stop else ("↑ escalated" if on else "· not needed")
        st.markdown(f'<div class="{cls}">{k} · {act}<br>'
                    f'<span style="font-size:10px">{hint} — {mark}</span></div>',
                    unsafe_allow_html=True)
    llm = trace.get("llm_calls", 0)
    tool = trace.get("tool_calls", 0)
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Retrieval calls", tool)
    c2.metric("LLM calls", llm, help="A static pipeline spends 1 on every query.")
    if llm == 0:
        st.caption("Resolved on cheap lanes — 0 LLM calls. Static would spend 1.")
    else:
        st.caption("Escalated to the rewrite: cheap lanes were insufficient.")


def render_citations(cites):
    for c in cites[:5]:
        tag = "spec" if c.get("source") == "spec" else "ts"
        tagtxt = "3GPP" if tag == "spec" else "scenario"
        st.markdown(
            f'<div class="cite"><span class="badge {tag}">{tagtxt}</span> '
            f'<b>{c.get("label","?")}</b><br>'
            f'<span style="color:#8296a8;font-size:12px">{(c.get("text") or "")[:240]}…</span>'
            f'</div>', unsafe_allow_html=True)


# ---- header + engine status --------------------------------------------------
st.markdown('<div class="sig">◢◤ 5G RAN · Rel-17 ◢◤</div>', unsafe_allow_html=True)
st.title("TelcoRAG")

health, herr = api_get("/health")
if herr:
    st.error(f"Engine unreachable at {API} — start it: `uvicorn serve:app --port 8000`  ({herr})")
elif health.get("status") == "ok":
    st.success(f"Engine online · {health['points']:,} points · {health['collection']}")
else:
    st.warning(f"Engine {health.get('status')}: {health.get('warning') or health.get('error')}")

srcs, _ = api_get("/sources")
with st.sidebar:
    st.markdown('<div class="sig">Corpus</div>', unsafe_allow_html=True)
    if srcs:
        st.write(f"**{srcs.get('scenario_chunks',0)}** troubleshooting chunks")
        specs = srcs.get("specs", {})
        for spec, n in sorted(specs.items(), key=lambda x: -x[1]):
            st.write(f"3GPP `{spec}` — {n}")
        st.caption(f"{srcs.get('total',0)} points total")
    st.markdown("---")
    st.caption(f"API: `{API}`")

tab_ask, tab_up, tab_rep, tab_tab, tab_dia = st.tabs(
    ["Ask", "Upload", "Report", "Table", "Diagram"])

# ---- ASK ---------------------------------------------------------------------
with tab_ask:
    st.markdown("Describe a 5G RAN symptom in plain language.")
    q = st.text_input("Query", key="askq",
                      placeholder="Calls from cars on the expressway die where one sector ends and the next begins…")
    if st.button("Diagnose", type="primary") and q.strip():
        main, side = st.columns([2, 1])
        with st.spinner("Climbing the diagnostic ladder…"):
            d, err = api_post("/ask", {"query": q})
        if err:
            st.error(err)
        else:
            with main:
                mode = d.get("generation_mode", "")
                st.markdown(f'<div class="sig">Diagnosis · {mode}</div>', unsafe_allow_html=True)
                st.write(d.get("answer", "(no answer)"))
                st.markdown('<div class="sig" style="margin-top:14px">Sources</div>', unsafe_allow_html=True)
                render_citations(d.get("citations", []))
            with side:
                render_ladder(d.get("trace", {}))

# ---- UPLOAD ------------------------------------------------------------------
with tab_up:
    st.markdown("Add your own telco documents to the corpus. "
                "PDF text is extracted in the browser-free path; plain text also works.")
    up = st.file_uploader("Upload a spec or notes file", type=["txt", "md", "pdf"])
    spec_hint = st.text_input("Spec label (optional, e.g. 38.331)", "")
    if up and st.button("Ingest into corpus"):
        # extract text: txt/md read directly; pdf needs PyMuPDF (server-side helper)
        if up.type == "application/pdf":
            try:
                import fitz  # PyMuPDF, same lib the ingest pipeline uses
                doc = fitz.open(stream=up.read(), filetype="pdf")
                text = "\n".join(p.get_text() for p in doc); doc.close()
            except Exception as e:
                text = ""
                st.error(f"PDF extraction failed ({e}). Install PyMuPDF: pip install pymupdf")
        else:
            text = up.read().decode("utf-8", errors="ignore")
        if text.strip():
            with st.spinner(f"Chunking + embedding {up.name}…"):
                r, err = api_post("/upload",
                                  {"name": up.name, "text": text,
                                   "spec": spec_hint or None}, timeout=300)
            if err:
                st.error(err)
            elif r.get("ok"):
                st.success(f"Ingested {up.name}: {r['chunks']} chunks added "
                           f"(spec={r.get('spec')}). Corpus updated — ask about it in the Ask tab.")
            else:
                st.error(r.get("error", "upload failed"))

# ---- REPORT / TABLE / DIAGRAM ------------------------------------------------
def artifact_tab(kind, blurb, render_as):
    st.markdown(blurb)
    q = st.text_input("Topic / query", key=f"{kind}q",
                     placeholder="e.g. too-late handover during fast mobility")
    if st.button(f"Generate {kind}", key=f"{kind}b", type="primary") and q.strip():
        with st.spinner(f"Retrieving + composing {kind}…"):
            d, err = api_post("/artifact", {"query": q, "kind": kind})
        if err:
            st.error(err); return
        content = d.get("content", "")
        if render_as == "markdown":
            st.markdown(content)
        elif render_as == "mermaid":
            # strip the ```mermaid fence and render via streamlit's mermaid support
            body = content.replace("```mermaid", "").replace("```", "").strip()
            st.markdown(f"```mermaid\n{body}\n```")
            with st.expander("Mermaid source"):
                st.code(body, language="text")
        st.markdown('<div class="sig" style="margin-top:12px">Grounded in</div>', unsafe_allow_html=True)
        render_citations(d.get("citations", []))
        tr = d.get("trace", {})
        st.caption(f"retrieval calls: {tr.get('tool_calls')} · LLM calls: {tr.get('llm_calls')} "
                   f"· stopped at step {tr.get('stop_step')}")

with tab_rep:
    artifact_tab("report", "A structured incident report, grounded and cited.", "markdown")
with tab_tab:
    artifact_tab("table", "Key parameters and timers as a sourced table.", "markdown")
with tab_dia:
    artifact_tab("diagram", "The fault mechanism as a Mermaid flowchart.", "mermaid")
