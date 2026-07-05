"""
agentic_rag_lab.py — A learner's agentic RAG playground.

Upload your own documents, then run the SAME question through five different
patterns and watch each one work, step by step:

  1. Plain RAG      — one search, one answer (the baseline)
  2. ReAct          — reason + act loop, can search several times
  3. Multi-hop      — chains retrievals: fact A -> fact B
  4. Reflection     — draft -> faithfulness critic -> re-retrieve if ungrounded
  5. Planning       — decompose into sub-questions, answer each, synthesise

Every pattern shows its full trace (reasoning, searches, chunks, critic verdict)
so you can SEE what each adds over plain RAG.

Run:  streamlit run agentic_rag_lab.py --server.address 0.0.0.0
Needs: OPENAI_API_KEY, the embedder on air, Qdrant reachable at localhost:6333.
"""

import io
import os
import uuid
import requests
import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
EMBED_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION = "learner_corpus"
DEFAULT_MODEL = "gpt-4o-mini"

client_oai = OpenAI()
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ----------------------------------------------------------------------------
# Retrieval backend (your pipeline) — the TOOL the agents call
# ----------------------------------------------------------------------------
def embed(text):
    r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]


def ensure_collection():
    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )


def search_documents(query, k=4):
    """The retrieval tool every pattern uses."""
    ensure_collection()
    hits = qdrant.query_points(
        collection_name=COLLECTION, query=embed(query), limit=k, with_payload=True,
    ).points
    return [{"source": h.payload["source"], "text": h.payload["text"],
             "score": round(h.score, 3)} for h in hits]


def chunk_text(text, size=600, overlap=100):
    words = text.split()
    chunks, cur, cur_len = [], [], 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len >= size:
            chunks.append(" ".join(cur))
            keep = cur[-(overlap // 6):] if overlap else []
            cur, cur_len = list(keep), sum(len(x) + 1 for x in keep)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    return data.decode("utf-8", errors="ignore")


def ingest_file(uploaded_file, batch_size=50):
    ensure_collection()
    text = extract_text(uploaded_file)
    if not text.strip():
        return 0
    chunks = chunk_text(text)
    total, batch = 0, []
    for i, ch in enumerate(chunks):
        batch.append(PointStruct(id=str(uuid.uuid4()), vector=embed(ch),
                     payload={"text": ch, "source": uploaded_file.name, "chunk_index": i}))
        if len(batch) >= batch_size:
            qdrant.upsert(collection_name=COLLECTION, points=batch); total += len(batch); batch = []
    if batch:
        qdrant.upsert(collection_name=COLLECTION, points=batch); total += len(batch)
    return total


# ----------------------------------------------------------------------------
# LLM brain
# ----------------------------------------------------------------------------
def call(messages, model=DEFAULT_MODEL):
    resp = client_oai.chat.completions.create(model=model, messages=messages, temperature=0.1)
    return resp.choices[0].message.content


def fmt_chunks(chunks):
    return "\n".join(f"[{c['source']}] {c['text']}" for c in chunks)


# ----------------------------------------------------------------------------
# The five patterns. Each is a GENERATOR that yields trace events so the UI
# can render them live. Event = dict with a "type": think|search|chunks|draft|
# critic|answer|plan|subq.
# ----------------------------------------------------------------------------
def pattern_plain_rag(question, model):
    """Baseline: one search, one answer. No loop, no judgment."""
    yield {"type": "think", "text": "Plain RAG: a single retrieval, then answer. No iteration."}
    chunks = search_documents(question)
    yield {"type": "search", "text": question}
    yield {"type": "chunks", "chunks": chunks}
    prompt = [
        {"role": "system", "content": "Answer using ONLY the context. If it's not there, say you don't know."},
        {"role": "user", "content": f"Context:\n{fmt_chunks(chunks)}\n\nQuestion: {question}"},
    ]
    yield {"type": "answer", "text": call(prompt, model),
           "sources": sorted({c["source"] for c in chunks})}


REACT_SYS = """You answer questions using a document search tool.
Each turn output EXACTLY ONE of:
  THINK: <reasoning>   (optional, one line, before an action)
  SEARCH: <query>
  ANSWER: <final answer, grounded ONLY in search results>
Output one action per turn then stop.

IMPORTANT stopping rules:
- After at most 2-3 searches you MUST give an ANSWER using the best information you have.
- Do NOT keep re-searching with slightly different wording — if results are similar to last time, stop and answer.
- Search using terms FROM THE QUESTION. Do not invent date constraints or assumptions the user didn't state.
- If the documents only partially answer, give the best grounded answer you can and note what's uncertain."""

def pattern_react(question, model, max_steps=6):
    """ReAct: reason+act loop. May search several times before answering."""
    messages = [{"role": "system", "content": REACT_SYS},
                {"role": "user", "content": f"Question: {question}"}]
    all_chunks = []
    for _ in range(max_steps):
        out = call(messages, model).strip()
        think = next((l.split(":", 1)[1].strip() for l in out.splitlines()
                      if l.strip().upper().startswith("THINK:")), "")
        if think:
            yield {"type": "think", "text": think}
        if "SEARCH:" in out:
            q = out.split("SEARCH:")[1].split("\n")[0].strip()
            chunks = search_documents(q)
            all_chunks.extend(chunks)
            yield {"type": "search", "text": q}
            yield {"type": "chunks", "chunks": chunks}
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": f"SEARCH RESULTS:\n{fmt_chunks(chunks)}\n\nContinue."})
        elif "ANSWER:" in out:
            yield {"type": "answer", "text": out.split("ANSWER:")[1].strip(), "sources": []}
            return
        else:
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": "Output SEARCH: or ANSWER:."})
    # step limit hit: FORCE a grounded answer from everything we retrieved
    yield {"type": "think", "text": "Step limit reached — forcing a grounded answer from all retrieved context."}
    seen, uniq = set(), []
    for c in all_chunks:
        if c["text"] not in seen:
            seen.add(c["text"]); uniq.append(c)
    forced = call([
        {"role": "system", "content": "Answer using ONLY the context below. Give the best grounded answer you can; note any uncertainty."},
        {"role": "user", "content": f"Context:\n{fmt_chunks(uniq)}\n\nQuestion: {question}"},
    ], model)
    yield {"type": "answer", "text": forced, "sources": sorted({c["source"] for c in uniq})}


MULTIHOP_SYS = """You answer questions that often require CHAINING searches:
search for fact A, then use what you learn to search for fact B, etc.
Each turn output EXACTLY ONE of:
  THINK: <what you know so far and what you still need>
  SEARCH: <query>
  ANSWER: <final answer grounded ONLY in results>
One action per turn, then stop.
Once you have the linked facts (usually after 2-3 searches), give the ANSWER.
Do not re-search with slightly different wording. Search using terms from the question; don't invent date assumptions."""

def pattern_multihop(question, model, max_steps=6):
    messages = [{"role": "system", "content": MULTIHOP_SYS},
                {"role": "user", "content": f"Question: {question}"}]
    hops = 0
    all_chunks = []
    for _ in range(max_steps):
        out = call(messages, model).strip()
        think = next((l.split(":", 1)[1].strip() for l in out.splitlines()
                      if l.strip().upper().startswith("THINK:")), "")
        if think:
            yield {"type": "think", "text": think}
        if "SEARCH:" in out:
            hops += 1
            q = out.split("SEARCH:")[1].split("\n")[0].strip()
            chunks = search_documents(q)
            all_chunks.extend(chunks)
            yield {"type": "search", "text": q, "hop": hops}
            yield {"type": "chunks", "chunks": chunks}
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": f"SEARCH RESULTS:\n{fmt_chunks(chunks)}\n\nContinue."})
        elif "ANSWER:" in out:
            yield {"type": "answer", "text": out.split("ANSWER:")[1].strip(),
                   "sources": [], "hops": hops}
            return
        else:
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": "Output SEARCH: or ANSWER:."})
    yield {"type": "think", "text": "Step limit reached — forcing an answer from all retrieved context."}
    seen, uniq = set(), []
    for c in all_chunks:
        if c["text"] not in seen:
            seen.add(c["text"]); uniq.append(c)
    forced = call([
        {"role": "system", "content": "Answer using ONLY the context below. Give the best grounded answer you can; note any uncertainty."},
        {"role": "user", "content": f"Context:\n{fmt_chunks(uniq)}\n\nQuestion: {question}"},
    ], model)
    yield {"type": "answer", "text": forced, "sources": sorted({c["source"] for c in uniq}), "hops": hops}


def pattern_reflection(question, model, max_rounds=3):
    """Draft -> faithfulness critic -> re-retrieve if ungrounded -> redraft."""
    query = question
    for rnd in range(max_rounds):
        yield {"type": "think", "text": f"Round {rnd+1}: retrieve, draft, then judge faithfulness."}
        chunks = search_documents(query)
        yield {"type": "search", "text": query}
        yield {"type": "chunks", "chunks": chunks}
        draft = call([
            {"role": "system", "content": "Answer using ONLY the context. If insufficient, say what's missing."},
            {"role": "user", "content": f"Context:\n{fmt_chunks(chunks)}\n\nQuestion: {question}"},
        ], model)
        yield {"type": "draft", "text": draft}
        # the faithfulness critic
        verdict = call([
            {"role": "system", "content": "You are a strict faithfulness judge. Given context and a draft "
             "answer, reply on the FIRST line with GROUNDED or NOT_GROUNDED, then one line explaining. "
             "NOT_GROUNDED if any claim isn't supported by the context."},
            {"role": "user", "content": f"Context:\n{fmt_chunks(chunks)}\n\nDraft answer: {draft}"},
        ], model)
        grounded = verdict.strip().upper().startswith("GROUNDED")
        yield {"type": "critic", "text": verdict, "grounded": grounded}
        if grounded:
            yield {"type": "answer", "text": draft, "sources": sorted({c["source"] for c in chunks})}
            return
        # not grounded: ask the model for a better search query and loop
        query = call([
            {"role": "system", "content": "The answer wasn't grounded. Propose a BETTER, more specific search "
             "query to find the missing evidence. Output only the query."},
            {"role": "user", "content": f"Question: {question}\nWhat we had: {fmt_chunks(chunks)[:500]}"},
        ], model).strip().strip('"')
        yield {"type": "think", "text": f"Not grounded — re-retrieving with: {query}"}
    yield {"type": "answer", "text": "(could not produce a grounded answer)", "sources": []}


def pattern_planning(question, model):
    """Decompose into sub-questions, answer each with retrieval, then synthesise."""
    plan = call([
        {"role": "system", "content": "Break the question into 2-4 concrete sub-questions, each answerable "
         "by a document search. Output one sub-question per line, no numbering."},
        {"role": "user", "content": question},
    ], model)
    subqs = [l.strip("-• ").strip() for l in plan.splitlines() if l.strip()][:4]
    yield {"type": "plan", "subqs": subqs}
    findings = []
    for sq in subqs:
        chunks = search_documents(sq)
        yield {"type": "subq", "text": sq}
        yield {"type": "chunks", "chunks": chunks}
        ans = call([
            {"role": "system", "content": "Answer the sub-question using ONLY the context. Be brief."},
            {"role": "user", "content": f"Context:\n{fmt_chunks(chunks)}\n\nSub-question: {sq}"},
        ], model)
        findings.append(f"Q: {sq}\nA: {ans}")
        yield {"type": "draft", "text": ans}
    final = call([
        {"role": "system", "content": "Synthesise the sub-answers into one coherent final answer."},
        {"role": "user", "content": f"Original question: {question}\n\nSub-answers:\n" + "\n\n".join(findings)},
    ], model)
    yield {"type": "answer", "text": final, "sources": []}


PATTERNS = {
    "Plain RAG (baseline — one search)": pattern_plain_rag,
    "ReAct (reason + act loop)": pattern_react,
    "Multi-hop (chain fact A → fact B)": pattern_multihop,
    "Reflection (faithfulness critic + re-retrieve)": pattern_reflection,
    "Planning (decompose → sub-answers → synthesise)": pattern_planning,
}

PATTERN_HELP = {
    "Plain RAG (baseline — one search)":
        "One retrieval, one answer. No loop, no self-check. Run this first, then run an agentic "
        "pattern on the SAME question to see what the agent adds.",
    "ReAct (reason + act loop)":
        "The model reasons, then acts (searches), sees the result, and decides again — looping until "
        "it can answer. The foundational agent pattern.",
    "Multi-hop (chain fact A → fact B)":
        "For questions where the answer needs a linking fact: find A, then use A to search for B. "
        "Watch the second query get built from the first result.",
    "Reflection (faithfulness critic + re-retrieve)":
        "Drafts an answer, then a critic judges whether every claim is grounded in the chunks. "
        "If NOT_GROUNDED, it re-retrieves with a better query and redrafts. The anti-hallucination pattern.",
    "Planning (decompose → sub-answers → synthesise)":
        "Breaks a complex question into sub-questions, answers each with its own retrieval, then "
        "synthesises. For broad questions one search can't cover.",
}


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Agentic RAG Lab", page_icon="🧪", layout="centered")

st.markdown("""
<style>
  .ev {padding:8px 14px; margin:8px 0; border-radius:6px; font-size:14px;}
  .e-think  {border-left:3px solid #BA7517; background:rgba(186,117,23,.06); font-style:italic; color:#854F0B;}
  .e-search {border-left:3px solid #185FA5; background:rgba(24,95,165,.06);}
  .e-search .q {font-family:monospace; color:#185FA5; font-weight:600;}
  .e-chunk  {font-size:12px; color:#5F5E5A; margin:2px 0 2px 14px; border-left:2px solid #D3D1C7; padding-left:8px;}
  .e-draft  {border-left:3px solid #888780; background:rgba(136,135,128,.06);}
  .e-ok     {border-left:3px solid #1D9E75; background:rgba(29,158,117,.08); font-weight:500; color:#0F6E56;}
  .e-bad    {border-left:3px solid #E24B4A; background:rgba(226,75,74,.08); font-weight:500; color:#A32D2D;}
  .e-final  {border-left:3px solid #639922; background:rgba(99,153,34,.10); padding:12px 16px; margin-top:12px;}
  .e-plan   {border-left:3px solid #534AB7; background:rgba(83,74,183,.06);}
</style>
""", unsafe_allow_html=True)

st.title("🧪 Agentic RAG Lab")
st.caption("Upload your documents · pick a pattern · watch the agent work · compare against plain RAG")

with st.sidebar:
    st.header("1 · Your documents")
    files = st.file_uploader("PDF, Word, or text", type=["pdf", "docx", "txt", "md"],
                             accept_multiple_files=True)
    if st.button("Ingest") and files:
        tot = 0
        for f in files:
            with st.spinner(f"Ingesting {f.name}…"):
                tot += ingest_file(f)
        st.success(f"Added {tot} chunks.")
    try:
        ensure_collection()
        st.metric("Chunks in corpus", qdrant.count(collection_name=COLLECTION).count)
    except Exception as e:
        st.error(f"Qdrant issue: {e}")
    if st.button("Clear corpus"):
        if qdrant.collection_exists(COLLECTION):
            qdrant.delete_collection(COLLECTION)
        st.rerun()
    st.divider()
    model = st.text_input("Model", value=DEFAULT_MODEL)

st.subheader("2 · Pick a pattern")
pattern_name = st.selectbox("Pattern", list(PATTERNS.keys()))
st.info(PATTERN_HELP[pattern_name])

st.subheader("3 · Ask")
question = st.text_input("Your question",
                         placeholder="Ask something about your uploaded documents…")

run = st.button("Run pattern", type="primary")

if run and question:
    st.markdown(f"#### Trace — {pattern_name.split(' (')[0]}")
    gen = PATTERNS[pattern_name](question, model)
    for ev in gen:
        t = ev["type"]
        if t == "think":
            st.markdown(f'<div class="ev e-think">💭 {ev["text"]}</div>', unsafe_allow_html=True)
        elif t == "plan":
            items = "".join(f"<li>{s}</li>" for s in ev["subqs"])
            st.markdown(f'<div class="ev e-plan"><b>Plan — sub-questions:</b><ul>{items}</ul></div>',
                        unsafe_allow_html=True)
        elif t == "subq":
            st.markdown(f'<div class="ev e-search">📋 sub-question: <span class="q">{ev["text"]}</span></div>',
                        unsafe_allow_html=True)
        elif t == "search":
            hop = f' (hop {ev["hop"]})' if "hop" in ev else ""
            st.markdown(f'<div class="ev e-search">🔍 search{hop}: <span class="q">{ev["text"]}</span></div>',
                        unsafe_allow_html=True)
        elif t == "chunks":
            html = "".join(f'<div class="e-chunk">[{c["source"]} · {c["score"]}] {c["text"][:130]}…</div>'
                           for c in ev["chunks"])
            st.markdown(html, unsafe_allow_html=True)
        elif t == "draft":
            st.markdown(f'<div class="ev e-draft">✍️ draft: {ev["text"]}</div>', unsafe_allow_html=True)
        elif t == "critic":
            cls = "e-ok" if ev["grounded"] else "e-bad"
            label = "✓ GROUNDED" if ev["grounded"] else "✗ NOT GROUNDED"
            st.markdown(f'<div class="ev {cls}">⚖️ critic: {label}<br><span style="font-weight:400">{ev["text"]}</span></div>',
                        unsafe_allow_html=True)
        elif t == "answer":
            extra = ""
            if "hops" in ev:
                extra = f' <span style="color:#3B6D11">· {ev["hops"]} hop(s)</span>'
            src = ev.get("sources") or []
            srctxt = f'<br><br><span style="font-size:12px;color:#5F5E5A">sources: {", ".join(src)}</span>' if src else ""
            st.markdown(f'<div class="ev e-final"><b>Answer</b>{extra}<br><br>{ev["text"]}{srctxt}</div>',
                        unsafe_allow_html=True)

    st.caption("Tip: run **Plain RAG** on this same question, then an agentic pattern, and compare the traces.")
