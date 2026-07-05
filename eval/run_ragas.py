"""
run_ragas.py - Rung 8: gold-free GENERATION evaluation (RAGAS-style).

Every rung so far measured RETRIEVAL (did the right chunk get fetched/ranked).
This rung measures the ANSWER the generator writes on top of that retrieval, on
the SAME frozen 22-query gold set, so the numbers sit next to the retrieval ladder.

The two headline metrics are GOLD-FREE - they need no reference answer, only the
question, the generated answer, and the retrieved context. That is the point: it
grades real user uploads for which no gold exists (the "what about no-gold" gap).

  FAITHFULNESS      of the claims the answer makes, how many are entailed by the
                    retrieved context.   =  supported_claims / total_claims
                    -> "is the answer grounded, or hallucinated?"   [0..1]

  ANSWER RELEVANCY  reverse the arrow: ask an LLM to invent the questions this
                    answer would answer, embed them, and measure cosine to the
                    ORIGINAL question.   =  mean cos(q, q_generated_i)   [0..1]
                    -> "does the answer address what was actually asked?"

Two REFERENCE-based extensions (opt-in, --with-reference) reuse the gold scenario
text (root_cause + resolution) as the reference:
  CONTEXT RECALL    of the reference's claims, how many are covered by the context.
  ANSWER CORRECTNESS overlap (F1 of claims) + semantic similarity to the reference.

Design decisions (each earns its place, per the project's rules):
  * Metrics are IMPLEMENTED, not imported from the ragas package. The project has
    always built its rulers (metrics.py, rrf.py) from scratch with self-tests, and
    a from-scratch impl (a) needs no ragas+langchain+datasets on a 12 GB box, (b)
    reuses the local Ollama judge + nomic-embed already running, (c) teaches the
    decomposition. Definitions follow Es et al., RAGAS (arXiv:2309.15217).
  * The JUDGE and the GENERATOR are switched INDEPENDENTLY (RAGAS_GEN / RAGAS_JUDGE,
    each 'local'|'openai', defaulting to config.GENERATOR). This is what makes the
    qwen-vs-gpt comparison clean AND exposes SELF-GRADING: if judge == generator the
    run is stamped self_grading=true and warns, because a model scoring its own
    answer is not an independent measurement.
  * Metric functions take injected `llm` and `embed` callables, so --selftest runs
    them on hand-traced mocks with NO services (same discipline as agent.py).
  * Every results file is config-stamped (gen, judge, embedder, k, prompt tags,
    corpus count) so runs made under different conditions are never silently
    compared - the habit that caught model drift twice.

Run (Qdrant server + Ollama up):
    python eval\\run_ragas.py --show                 # local gen + local judge (fast)
    # frontier judge, but keep it FAST (entailment needs no reasoning model):
    $env:RAGAS_GEN="local"; $env:RAGAS_JUDGE="openai"; $env:RAGAS_JUDGE_MODEL="gpt-4o-mini"
    python eval\\run_ragas.py --show
    python eval\\run_ragas.py --limit 3 --show        # sanity-check speed on 3 queries first
    python eval\\run_ragas.py --with-reference        # + context-recall/correctness (slower)
    python eval\\run_ragas.py --selftest              # metrics on mocks, no services

Speed note: with RAGAS_JUDGE=openai the judge model matters enormously. gpt-5.5 is a
REASONING model (~10-40s/call) -> a full --with-reference run is 200+ calls and can take
an hour. Set RAGAS_JUDGE_MODEL=gpt-4o-mini for a ~10x speedup; the judge only does
entailment/usefulness verdicts, which a small model does well. Progress now prints per
query so a slow run is never a silent wait, and Ctrl-C is safe.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "eval" / "gold_set_v1.jsonl"
TS = ROOT / "corpus" / "troubleshooting.jsonl"

# Prompt tags recorded in every results file: change one -> runs are not comparable.
CLAIM_TAG = "claim-v1"
VERDICT_TAG = "verdict-v1"
QGEN_TAG = "qgen-v1"

TIERS = ("easy", "medium", "hard")

# --------------------------------------------------------------------------
# Metric core - each takes injected llm(prompt)->str and embed(text)->vec so the
# self-test can drive them with mocks and zero services.
# --------------------------------------------------------------------------
def _lines(text):
    """Split an LLM list-reply into clean, de-bulleted, non-empty lines."""
    out = []
    for ln in (text or "").splitlines():
        s = ln.strip().lstrip("-*0123456789.) ").strip()
        if s:
            out.append(s)
    return out


def extract_claims(text, llm):
    """Decompose a passage into atomic factual claims (one assertion each)."""
    prompt = (
        "Break the text into a list of atomic factual claims - each a single, "
        "self-contained assertion, no pronouns. One claim per line, no numbering.\n\n"
        f"Text:\n{text}\n\nClaims:"
    )
    claims = _lines(llm(prompt))
    return [c for c in claims if len(c) > 3]


def _verdicts(claims, context, llm):
    """For each claim, 1 if entailed by context else 0. One LLM call for all."""
    if not claims:
        return []
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    prompt = (
        "For each numbered claim, decide if it can be inferred from the CONTEXT. "
        "Reply with the claim number and 1 (supported) or 0 (not supported), one "
        "per line, e.g. '1: 1'. Judge ONLY from the context.\n\n"
        f"CONTEXT:\n{context}\n\nCLAIMS:\n{numbered}\n\nVerdicts:"
    )
    reply = llm(prompt)
    verdicts = {}
    for ln in reply.splitlines():
        m = re.match(r"\s*(\d+)\s*[:).\-]\s*([01])", ln)
        if m:
            verdicts[int(m.group(1))] = int(m.group(2))
    # default any un-parsed claim to 0 (unsupported) - the conservative choice.
    return [verdicts.get(i + 1, 0) for i in range(len(claims))]


def faithfulness(answer, contexts, llm):
    """supported_claims / total_claims, judged against the retrieved context."""
    ctx = "\n\n".join(contexts)
    claims = extract_claims(answer, llm)
    if not claims:
        return 0.0, {"claims": 0, "supported": 0}
    v = _verdicts(claims, ctx, llm)
    supported = sum(v)
    return supported / len(claims), {"claims": len(claims), "supported": supported}


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


_NONCOMMITTAL = re.compile(
    r"\b(i (don'?t|do not) know|cannot (be )?answer|no (relevant )?(context|information)|"
    r"unable to|not enough (info|context))\b", re.IGNORECASE)


def answer_relevancy(question, answer, llm, embed, n=3):
    """Reverse-question generation + cosine to the original question.
    Non-committal answers score 0 (they do not address the question)."""
    if not answer or _NONCOMMITTAL.search(answer):
        return 0.0, {"generated": 0, "noncommittal": True}
    prompt = (
        f"Given the ANSWER below, write {n} distinct questions that this answer "
        "would correctly and directly answer. One question per line, no numbering.\n\n"
        f"ANSWER:\n{answer}\n\nQuestions:"
    )
    gen_qs = _lines(llm(prompt))[:n]
    if not gen_qs:
        return 0.0, {"generated": 0, "noncommittal": False}
    qv = embed(question)
    sims = [max(0.0, _cos(qv, embed(g))) for g in gen_qs]
    return sum(sims) / len(sims), {"generated": len(gen_qs), "noncommittal": False}


def context_precision(question, answer, contexts, llm):
    """Rank-weighted average precision of USEFUL contexts (reference-free).
    ONE judge call scores ALL chunks at once - not one call per chunk. This is the
    fix that keeps context-precision affordable when the judge is a slow reasoning
    model (was k calls/query; now 1)."""
    if not contexts:
        return 0.0
    numbered = "\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts))
    prompt = (
        "For each numbered CONTEXT, decide if it helps answer the QUESTION. Reply "
        "with the number and 1 (useful) or 0 (not useful), one per line, e.g. '1: 1'.\n\n"
        f"QUESTION: {question}\n\nCONTEXTS:\n{numbered}\n\nVerdicts:"
    )
    reply = llm(prompt)
    verd = {}
    for ln in reply.splitlines():
        mm = re.match(r"\s*(\d+)\s*[:).\-]\s*([01])", ln)
        if mm:
            verd[int(mm.group(1))] = int(mm.group(2))
    rels = [verd.get(i + 1, 0) for i in range(len(contexts))]
    total_rel = sum(rels)
    if total_rel == 0:
        return 0.0
    running, ap = 0, 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            running += 1
            ap += running / i
    return ap / total_rel


def context_recall(reference, contexts, llm):
    """Of the reference's claims, how many are attributable to the context."""
    ctx = "\n\n".join(contexts)
    ref_claims = extract_claims(reference, llm)
    if not ref_claims:
        return 0.0
    v = _verdicts(ref_claims, ctx, llm)
    return sum(v) / len(ref_claims)


def answer_correctness(answer, reference, llm, embed):
    """F1 over claim overlap + semantic similarity, averaged (RAGAS blends both)."""
    a_claims = extract_claims(answer, llm)
    r_claims = extract_claims(reference, llm)
    if not a_claims or not r_claims:
        f1 = 0.0
    else:
        # a claim is a "match" if entailed by the reference blob, and vice versa.
        tp = sum(_verdicts(a_claims, reference, llm))          # answer claims in ref
        fn = len(r_claims) - sum(_verdicts(r_claims, "\n".join(a_claims), llm))
        fp = len(a_claims) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + max(fn, 0)) if (tp + max(fn, 0)) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    sim = max(0.0, _cos(embed(answer), embed(reference)))
    return 0.5 * f1 + 0.5 * sim


# --------------------------------------------------------------------------
# Real backends (Ollama local / OpenAI) - mirror rewrite.py exactly.
# --------------------------------------------------------------------------
def _backend(kind):
    """kind='gen'|'judge'. RAGAS_GEN / RAGAS_JUDGE override config.GENERATOR."""
    env = {"gen": "RAGAS_GEN", "judge": "RAGAS_JUDGE"}[kind]
    return (os.getenv(env) or config.GENERATOR or "local").lower()


def _eff_model(kind):
    """Effective model name. A per-role override lets the JUDGE use a FAST model
    (e.g. gpt-4o-mini) even when config/generator uses gpt-5.5: entailment does not
    need a reasoning model, and gpt-5.5 is ~10-40s/call.
        RAGAS_GEN_MODEL   overrides the generator's model
        RAGAS_JUDGE_MODEL overrides the judge's model
    """
    override = os.getenv({"gen": "RAGAS_GEN_MODEL", "judge": "RAGAS_JUDGE_MODEL"}[kind])
    if _backend(kind) == "openai":
        return override or config.OPENAI_MODEL or "gpt-4o-mini"
    return override or config.LOCAL_GEN_MODEL


def _model_label(kind):
    return ("openai:" if _backend(kind) == "openai" else "ollama:") + _eff_model(kind)


def _make_llm(kind):
    """Return an llm(prompt)->str closure for the chosen backend, with retry+backoff
    on transient OpenAI errors (429/5xx) so one blip never kills a long run."""
    import time as _t
    import urllib.error
    import urllib.request
    model = _eff_model(kind)

    def _ollama(prompt):
        body = json.dumps({"model": model, "prompt": prompt,
                           "stream": False, "options": {"temperature": 0.0}}).encode()
        req = urllib.request.Request(config.OLLAMA_URL + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode()).get("response", "").strip()

    def _openai(prompt):
        key = config.OPENAI_API_KEY or ""
        if not key or "REPLACE" in key:
            raise SystemExit("OPENAI_API_KEY not set (needed for RAGAS_%s=openai)" % kind)
        body = json.dumps({"model": model,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions", data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                    _t.sleep(2 ** attempt)          # 1s, 2s, 4s
                    continue
                raise
        return ""

    return _openai if _backend(kind) == "openai" else _ollama


GEN_SYS = ("You are a 5G RAN troubleshooting assistant. Answer the question using "
           "ONLY the numbered context. Cite sources as [n]. If the context does not "
           "contain the answer, say so - do not invent parameters.")


def generate_answer(question, contexts, gen_llm):
    ctx = "\n\n".join(f"[{i}] {c[:600]}" for i, c in enumerate(contexts, 1))
    prompt = f"{GEN_SYS}\n\nContext:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    return gen_llm(prompt)


# --------------------------------------------------------------------------
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def reference_for(gold_ids, ts_by_id):
    """Assemble a reference answer from the gold scenario(s): root_cause + resolution."""
    parts = []
    for gid in gold_ids:
        s = ts_by_id.get(gid)
        if s:
            parts.append(f"{s.get('root_cause','')} {s.get('resolution','')}".strip())
    return " ".join(parts)


def aggregate(records, keys):
    def mean(rows, k):
        vals = [r[k] for r in rows if r.get(k) is not None]
        return sum(vals) / len(vals) if vals else None
    summary = {"overall": {k: mean(records, k) for k in keys}}
    summary["overall"]["n"] = len(records)
    summary["by_tier"] = {}
    for t in TIERS:
        rows = [r for r in records if r["tier"] == t]
        summary["by_tier"][t] = {**{k: mean(rows, k) for k in keys}, "n": len(rows)}
    return summary


def fmt(summary, keys, title):
    def row(name, d):
        cells = "  ".join(f"{k}={d[k]:.3f}" if d.get(k) is not None else f"{k}=  -  "
                          for k in keys)
        return f"  {name:<8} n={d.get('n',0):<3} {cells}"
    lines = [f"=== {title} ===", row("overall", summary["overall"])]
    for t in TIERS:
        lines.append(row(t, summary["by_tier"][t]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="contexts passed to the generator")
    ap.add_argument("--lane-k", type=int, default=10, dest="lane_k")
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--ce-floor", type=float, default=0.5, dest="ce_floor")
    ap.add_argument("--n-questions", type=int, default=3, dest="nq")
    ap.add_argument("--with-reference", action="store_true",
                    help="also compute context-recall + answer-correctness vs gold text")
    ap.add_argument("--limit", type=int, default=0, help="first N queries only (debug)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    # ---- heavy imports here so --selftest needs no services ----
    from store import get_client
    from embed import embed_one
    from bm25_tool import ensure_index
    from rerank import rerank
    from rewrite import rewrite_query
    from agent import Agent

    gold = load_jsonl(GOLD)
    if args.limit:
        gold = gold[:args.limit]
    ts_by_id = {r["id"]: r for r in load_jsonl(TS)}

    client = get_client()
    corpus_n = client.count(config.COLLECTION, exact=True).count
    bm25 = ensure_index(client)
    tools = {"dense": lambda q, k: client.query_points(config.COLLECTION, query=embed_one(q),
                                                       limit=k, with_payload=True).points,
             "bm25": lambda q, k: bm25.search(q, k=k),
             "rewrite": rewrite_query, "rerank": rerank}
    agent = Agent(tools, k=args.k, lane_k=args.lane_k, budget=args.budget,
                  ce_floor=args.ce_floor)

    gen_llm = _make_llm("gen")
    judge_llm = _make_llm("judge")
    self_grading = _model_label("gen") == _model_label("judge")
    if self_grading:
        print("WARNING: judge == generator -> SELF-GRADING. A model scoring its own\n"
              "         answer is not an independent measurement. Set RAGAS_JUDGE to a\n"
              "         different backend for a trustworthy faithfulness number.\n")

    core_keys = ["faithfulness", "answer_relevancy", "context_precision"]
    ref_keys = ["context_recall", "answer_correctness"] if args.with_reference else []
    all_keys = core_keys + ref_keys

    # up-front cost estimate + reasoning-judge warning, so a slow judge is never a
    # silent 20-minute wait again.
    per_q = 3 + (6 if args.with_reference else 0)      # faithfulness+ctxprec (+ref)
    print(f"\ngen={_model_label('gen')}   judge={_model_label('judge')}   "
          f"self_grading={self_grading}   queries={len(gold)}")
    if _backend("judge") == "openai":
        jm = _eff_model("judge")
        print(f"~{per_q * len(gold)} judge API calls (~{per_q}/query). ", end="")
        if re.search(r"(gpt-5|o1|o3)", jm):
            print(f"\n  NOTE: '{jm}' is a REASONING model (~10-40s/call) -> this run can\n"
                  "        take an HOUR+. For ~10x speedup (entailment needs no reasoning):\n"
                  "            $env:RAGAS_JUDGE_MODEL=\"gpt-4o-mini\"\n"
                  "        Or sanity-check speed first with:  --limit 3 --show")
        else:
            print("(fast judge - good.)")
    print("  progress prints per query below; Ctrl-C is safe (partial run discarded).\n")

    import time as _time
    records = []
    for i, g in enumerate(gold, 1):
        t0 = _time.time()
        try:
            tr = agent.run(g["qid"], g["query"])
            contexts = [(h.payload or {}).get("text", "") for h in tr.final_hits[:args.k]]
            contexts = [c for c in contexts if c]
            answer = generate_answer(g["query"], contexts, gen_llm)

            f_score, f_meta = faithfulness(answer, contexts, judge_llm)
            ar_score, _ = answer_relevancy(g["query"], answer, gen_llm, embed_one, n=args.nq)
            cp_score = context_precision(g["query"], answer, contexts, judge_llm)
            rec = {"qid": g["qid"], "tier": g["difficulty"],
                   "faithfulness": f_score, "answer_relevancy": ar_score,
                   "context_precision": cp_score,
                   "n_claims": f_meta["claims"], "n_supported": f_meta["supported"],
                   "llm_calls": tr.llm_calls, "tool_calls": tr.tool_calls}
            if args.with_reference:
                ref = reference_for(g["relevant"], ts_by_id)
                rec["context_recall"] = context_recall(ref, contexts, judge_llm) if ref else None
                rec["answer_correctness"] = answer_correctness(answer, ref, judge_llm, embed_one) if ref else None
            records.append(rec)

            dt = _time.time() - t0
            print(f"[{i:2d}/{len(gold)}] {g['qid']} {g['difficulty']:<6} "
                  f"faith={f_score:.2f} relev={ar_score:.2f} ctxP={cp_score:.2f}  "
                  f"({f_meta['supported']}/{f_meta['claims']} claims, {dt:.0f}s)", flush=True)
            if args.show:
                print(f"        A: {answer[:180]}", flush=True)
        except KeyboardInterrupt:
            print("\ninterrupted - partial run discarded (nothing saved). Re-run when ready.")
            return
        except Exception as e:
            print(f"[{i:2d}/{len(gold)}] {g['qid']} ERROR: {type(e).__name__}: "
                  f"{str(e)[:120]} -> skipped", flush=True)
            continue

    if not records:
        die("no queries scored - check the errors above.")

    summary = aggregate(records, all_keys)
    print("\n" + fmt(summary, all_keys,
                     f"RAGAS gen={_model_label('gen')} judge={_model_label('judge')}"))
    print("\nheadline (gold-free): FAITHFULNESS + ANSWER RELEVANCY are the two that "
          "need no reference; the rest are diagnostic.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"run": "ragas", "timestamp": stamp,
               "config": {"generator": _model_label("gen"),
                          "judge": _model_label("judge"),
                          "self_grading": self_grading,
                          "embedder": config.EMBED_MODEL,
                          "k": args.k, "lane_k": args.lane_k, "budget": args.budget,
                          "n_questions": args.nq, "with_reference": args.with_reference,
                          "qdrant_mode": config.QDRANT_MODE, "corpus_count": corpus_n,
                          "claim_tag": CLAIM_TAG, "verdict_tag": VERDICT_TAG,
                          "qgen_tag": QGEN_TAG},
               "summary": summary, "per_query": records}
    out = ROOT / "eval" / f"results_ragas_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out.name}")


# --------------------------------------------------------------------------
# Self-test: mock judge + mock embed, hand-traced metric values, then asserted.
# No Qdrant, no Ollama, no OpenAI.
# --------------------------------------------------------------------------
def _selftest():
    print("run_ragas self-test (mock judge + embed, hand-traced)\n" + "=" * 60)

    # A mock judge that: (extract) splits sentences on '. '; (verdict) supports a
    # claim iff the word 'GROUNDED' appears in it; (question-gen) echoes 2 lines;
    # (useful) 1 if 'ctxok' in the context. We drive each metric to a known value.
    def judge(prompt):
        p = prompt.lower()
        if "atomic factual claims" in p:
            body = prompt.split("Text:\n", 1)[1].split("\n\nClaims:")[0]
            return "\n".join(s.strip() for s in body.split(". ") if s.strip())
        if "for each numbered claim" in p:
            claims = re.findall(r"\d+\.\s*(.+)", prompt.split("CLAIMS:")[1])
            return "\n".join(f"{i+1}: {1 if 'grounded' in c.lower() else 0}"
                             for i, c in enumerate(claims))
        if "for each numbered context" in p:
            block = prompt.split("CONTEXTS:", 1)[1]
            items = re.findall(r"\[(\d+)\]\s*([^\[]+)", block)
            return "\n".join(f"{num}: {1 if 'ctxok' in txt.lower() else 0}"
                             for num, txt in items)
        return ""

    # deterministic mock embedder: bag-of-words hashed into a small vector.
    def embed(text):
        v = [0.0] * 16
        for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
            v[hash(w) % 16] += 1.0
        return v

    # 1) faithfulness: 2 grounded claims + 1 not -> 2/3.
    ans = "Alpha is GROUNDED here. Beta is GROUNDED too. Gamma is invented."
    f, meta = faithfulness(ans, ["context"], judge)
    print(f"faithfulness = {f:.3f}  (expect 0.667)  meta={meta}")
    assert abs(f - 2 / 3) < 1e-6 and meta == {"claims": 3, "supported": 2}

    # 2) faithfulness of a fully-grounded answer -> 1.0; fully-invented -> 0.0.
    assert faithfulness("X is GROUNDED. Y is GROUNDED.", ["c"], judge)[0] == 1.0
    assert faithfulness("X invented. Y invented.", ["c"], judge)[0] == 0.0

    # 3) answer relevancy: identical text embeds to itself -> cosine 1.0. Mock
    #    question-gen returns the question twice, so relevancy = 1.0.
    def judge_qgen(prompt):
        if "distinct questions" in prompt.lower():
            return "ping pong handover\nping pong handover"
        return judge(prompt)
    ar, arm = answer_relevancy("ping pong handover", "some grounded answer text",
                               judge_qgen, embed, n=2)
    print(f"answer_relevancy = {ar:.3f}  (expect 1.000)")
    assert abs(ar - 1.0) < 1e-6

    # 4) non-committal answer -> relevancy 0.0.
    ar0, _ = answer_relevancy("q", "I don't know from the context.", judge_qgen, embed)
    assert ar0 == 0.0

    # 5) context precision: useful chunk first (ctxok) then junk -> AP = 1.0/1 = 1.0.
    cp = context_precision("q", "a", ["ctxok chunk", "junk chunk"], judge)
    print(f"context_precision = {cp:.3f}  (expect 1.000)")
    assert abs(cp - 1.0) < 1e-6
    # useful chunk SECOND -> precision at rank 2 = 1/2 -> AP = 0.5.
    cp2 = context_precision("q", "a", ["junk chunk", "ctxok chunk"], judge)
    print(f"context_precision (relevant 2nd) = {cp2:.3f}  (expect 0.500)")
    assert abs(cp2 - 0.5) < 1e-6

    # 6) cosine sanity: orthogonal -> 0, identical -> 1.
    assert abs(_cos([1, 0], [0, 1])) < 1e-9
    assert abs(_cos([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9

    print("\nall asserts passed - RAGAS metric core behaves as hand-traced.")


if __name__ == "__main__":
    main()
