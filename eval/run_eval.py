"""
run_eval.py - score the gold set through retrieval and record the result.

Modes on the SAME 22 queries, so comparisons are honest:
  (default)   plain dense retrieval           -> the baseline
  --rerank    dense top-K, cross-encoder reorders it
  --rewrite   dense(original) + dense(qwen-rewrite), merge, then rerank with the
              ORIGINAL query. The rewrite is a retrieval aid; relevance is still
              judged against what the user actually asked. Because re-fetching a
              rewritten query changes the candidate set, hard-tier R@10 can rise
              here - the thing reranking alone could not do.
  --hybrid    dense top-K + BM25 top-K fused by Reciprocal Rank Fusion (rung 7b).
              Scores never cross lanes; ranks do. RRF picks WHICH candidates
              form the pool; adding --rerank lets the cross-encoder order that
              pool. Lane depth defaults to --k so the comparison against the
              dense baseline is not confounded by fetch depth (override with
              --lane-k for depth experiments). Combined with --rewrite (rung 7c)
              the rewrite feeds BOTH lanes: four lists fused - dense(orig),
              dense(rw), bm25(orig), bm25(rw) - RRF shortlists 2K (a single-lane
              champion like TS-01-via-bm25(rw) must survive to the judge), then
              the cross-encoder orders the pool on the ORIGINAL query.

Run (Qdrant server + Ollama up):
  python eval\\run_eval.py                     # baseline
  python eval\\run_eval.py --rerank            # + cross-encoder
  python eval\\run_eval.py --rewrite --show    # + query rewrite (needs qwen)
"""
import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

# eval\ (this file's directory) is already sys.path[0]. APPEND the project root
# so store/embed/config resolve from root, WITHOUT letting a stray root-level
# copy of an eval module (e.g. a duplicate rewrite.py) shadow the real one here.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from store import get_client
from embed import embed_one
import config
import metrics

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "eval" / "gold_set_v1.jsonl"
OUTDIR = ROOT / "eval"


def hit_label(payload: dict) -> str:
    if payload.get("id"):
        return payload["id"]
    return f"spec:{payload.get('spec', '?')} {payload.get('clause', '?')}"


def load_gold():
    rows = []
    with open(GOLD, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_hits(*lists):
    """Union of candidate hits, deduped by point id, order preserved."""
    seen = {}
    for lst in lists:
        for h in lst:
            if h.id not in seen:
                seen[h.id] = h
    return list(seen.values())


def latest_results(tag):
    files = sorted(glob.glob(str(OUTDIR / f"results_{tag}_*.json")))
    return json.loads(Path(files[-1]).read_text(encoding="utf-8")) if files else None


def print_lift(base, new, base_name):
    bo, no = base["summary"]["overall"], new["summary"]["overall"]
    print(f"\n=== lift over {base_name} ===")
    print(f"{'slice':<12}{'MRR':>18}{'R@5':>18}{'R@10':>18}")

    def line(name, b, n):
        def cell(k):
            d = n[k] - b[k]
            return f"{b[k]:.3f}->{n[k]:.3f} ({'+' if d >= 0 else ''}{d:.3f})"
        print(f"{name:<12}{cell('mrr'):>18}{cell('r@5'):>18}{cell('r@10'):>18}")

    line("all", bo, no)
    for tier in ("easy", "medium", "hard"):
        b = base["summary"]["by_tier"].get(tier)
        n = new["summary"]["by_tier"].get(tier)
        if b and n:
            line(tier, b, n)
    print("watch: hard R@10 - only a NEW FETCH (rewrite or a second lane) can "
          "raise it; rerank alone cannot.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=max(metrics.KS),
                    help="retrieval depth per query (default = largest cutoff, 10)")
    ap.add_argument("--rerank", action="store_true",
                    help="cross-encoder reorders the dense top-K")
    ap.add_argument("--rewrite", action="store_true",
                    help="add a qwen query rewrite, merge candidates, then rerank")
    ap.add_argument("--hybrid", action="store_true",
                    help="fuse dense + BM25 lanes with RRF (rung 7b)")
    ap.add_argument("--lane-k", type=int, default=None, dest="lane_k",
                    help="fetch depth per lane in hybrid mode (default: --k)")
    ap.add_argument("--ce-floor", type=float, default=0.0, dest="ce_floor",
                    help="confidence gate (hybrid modes): if the cross-encoder's "
                         "best score over the pool is below this, the judge has "
                         "no opinion - keep the RRF order. 0.0 = always trust "
                         "the judge (previous behaviour). Autopsy: confident "
                         "~0.99, collapsed ~0.000; 0.5 is the natural gate.")
    ap.add_argument("--pool-mult", type=int, default=2, dest="pool_mult",
                    help="hybrid+rewrite shortlist = pool_mult*K candidates to "
                         "the reranker (default 2; 4 = the full 4-lane union, "
                         "guaranteeing single-lane champions reach the judge)")
    ap.add_argument("--show", action="store_true",
                    help="print each query's ranked hits (and the rewrite)")
    args = ap.parse_args()

    gold = load_gold()
    client = get_client()

    bm25_index = rrf_fuse = None
    if args.hybrid:
        from rrf import rrf_fuse, RRF_K
        from bm25_tool import ensure_index
        bm25_index = ensure_index(client)   # fingerprint-checked vs live count

    rerank_fn = rewrite_fn = None
    if args.rerank or args.rewrite:
        from rerank import rerank as rerank_fn
    if args.rewrite:
        from rewrite import rewrite_query as rewrite_fn
        from rewrite import active_generator, PROMPT_TAG
        gen_label = active_generator()

    def search(vec, limit=None):
        return client.query_points(config.COLLECTION, query=vec,
                                   limit=limit or args.k,
                                   with_payload=True).points

    records = []
    for g in gold:
        rw = None
        ce_max = gate = None
        if args.rewrite and args.hybrid:
            rw = rewrite_fn(g["query"])
            lane_k = args.lane_k or args.k
            lanes = {
                "do": search(embed_one(g["query"]), limit=lane_k),
                "dr": search(embed_one(rw), limit=lane_k),
                "bo": bm25_index.search(g["query"], k=lane_k),
                "br": bm25_index.search(rw, k=lane_k),
            }
            lane_ids = {n: {h.id for h in lst} for n, lst in lanes.items()}
            fused = rrf_fuse(*lanes.values())
            pool = [h for h, _ in fused[:args.pool_mult * args.k]]  # widen so
            # single-lane champions (one vote ~0.016 vs coalitions >=0.029)
            # can still reach the judge; --pool-mult 4 = the full union
            judged = rerank_fn(g["query"], pool)
            ce_max = max((sc for _, sc in judged), default=0.0)
            gate = ce_max < args.ce_floor
            reordered = fused[:args.k] if gate else judged[:args.k]
            disp = []
            for h, s in reordered:
                prov = "+".join(n for n in ("do", "dr", "bo", "br")
                                if h.id in lane_ids[n])
                disp.append((hit_label(h.payload or {}), s, prov))
        elif args.rewrite:
            rw = rewrite_fn(g["query"])
            hits_o = search(embed_one(g["query"]))
            hits_r = search(embed_one(rw))
            ids_o = {h.id for h in hits_o}
            ids_r = {h.id for h in hits_r}
            pool = merge_hits(hits_o, hits_r)
            reordered = rerank_fn(g["query"], pool)          # rerank on ORIGINAL query
            disp = []
            for h, s in reordered[:args.k]:
                prov = ("both" if h.id in ids_o and h.id in ids_r
                        else "rw" if h.id in ids_r else "orig")
                disp.append((hit_label(h.payload or {}), s, prov))
        elif args.hybrid:
            lane_k = args.lane_k or args.k
            hits_d = search(embed_one(g["query"]), limit=lane_k)
            hits_b = bm25_index.search(g["query"], k=lane_k)
            ids_d = {h.id for h in hits_d}
            ids_b = {h.id for h in hits_b}
            fused = rrf_fuse(hits_d, hits_b)             # ranks, never scores
            pool = [h for h, _ in fused[:args.k]]        # RRF picks the pool
            if args.rerank:
                judged = rerank_fn(g["query"], pool)
                ce_max = max((sc for _, sc in judged), default=0.0)
                gate = ce_max < args.ce_floor
                reordered = fused[:args.k] if gate else judged
            else:
                reordered = fused[:args.k]
            disp = []
            for h, s in reordered:
                prov = ("both" if h.id in ids_d and h.id in ids_b
                        else "bm25" if h.id in ids_b else "dense")
                disp.append((hit_label(h.payload or {}), s, prov))
        else:
            hits = search(embed_one(g["query"]))
            if args.rerank:
                reordered = rerank_fn(g["query"], hits)
                disp = [(hit_label(h.payload or {}), s, "") for h, s in reordered]
            else:
                disp = [(hit_label(h.payload or {}), h.score, "") for h in hits]

        ranked = [lbl for lbl, _, _ in disp]
        s = metrics.score_query(ranked, g["relevant"])
        rec = {"qid": g["qid"], "tier": g["difficulty"], **s}
        if rw is not None:
            rec["rewrite"] = rw
        if ce_max is not None:
            rec["ce_max"] = round(ce_max, 4)
            rec["gate"] = bool(gate)
        records.append(rec)

        if args.show:
            rel = set(g["relevant"])
            print(f"\n{g['qid']} [{g['difficulty']}]  gold={g['relevant']}")
            print(f"  {g['query']}")
            if rw is not None:
                print(f"  rewrite-> {rw}")
            if ce_max is not None:
                print(f"  judge: max score {ce_max:.4f} -> "
                      f"{'GATED, RRF order kept' if gate else 'CE order used'}")
            for i, (lbl, sc, prov) in enumerate(disp, 1):
                mark = "*" if lbl in rel else " "
                tag = f" ({prov})" if prov else ""
                print(f"  {mark}{i:>2}. [{sc:+.3f}] {lbl}{tag}")
            print(f"  -> P@5={s['p@5']:.3f} R@5={s['r@5']:.3f} "
                  f"R@10={s['r@10']:.3f} RR={s['rr']:.3f}")

    summary = metrics.aggregate(records)
    mode = ("hybrid RRF + rewrite + rerank" if args.rewrite and args.hybrid
            else "dense+rewrite+rerank" if args.rewrite
            else "hybrid RRF + rerank" if args.hybrid and args.rerank
            else "hybrid RRF (dense+BM25)" if args.hybrid
            else "dense+rerank" if args.rerank else "plain dense retrieval")
    print("\n" + metrics.format_table(summary, title=mode))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = ("hybridrewrite" if args.rewrite and args.hybrid
           else "rewrite" if args.rewrite
           else "hybridrerank" if args.hybrid and args.rerank
           else "hybrid" if args.hybrid
           else "rerank" if args.rerank else "baseline")
    run_name = ("hybrid_rrf_rewrite_rerank" if args.rewrite and args.hybrid
                else "dense_rewrite_rerank" if args.rewrite
                else "hybrid_rrf_rerank" if args.hybrid and args.rerank
                else "hybrid_rrf" if args.hybrid
                else "dense_plus_rerank" if args.rerank else "plain_dense_retrieval")
    out = OUTDIR / f"results_{tag}_{stamp}.json"
    payload = {
        "run": run_name,
        "timestamp": stamp,
        "config": {
            "embed_model": config.EMBED_MODEL,
            "collection": config.COLLECTION,
            "qdrant_mode": config.QDRANT_MODE,
            "retrieval_k": args.k,
            "reranker": config.RERANK_MODEL if (args.rerank or args.rewrite) else None,
            "generator": gen_label if args.rewrite else None,
            "rewrite_prompt": PROMPT_TAG if args.rewrite else None,
            "hybrid": args.hybrid,
            "lane_k": (args.lane_k or args.k) if args.hybrid else None,
            "rrf_k": RRF_K if args.hybrid else None,
            "ce_floor": args.ce_floor if args.hybrid else None,
            "pool_k": (args.pool_mult * args.k if (args.hybrid and args.rewrite)
                       else args.k if args.hybrid else None),
            "cutoffs": list(metrics.KS),
        },
        "gold_set": GOLD.name,
        "summary": summary,
        "per_query": records,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out.name}")

    if args.rewrite and args.hybrid:
        shown = False
        for tag_, nm in (("hybridrerank", "hybrid+rerank"), ("rewrite", "rewrite")):
            base = latest_results(tag_)
            if base:
                print_lift(base, payload, nm)
                shown = True
        if not shown:
            print("(no prior results to compare against.)")
    elif args.rewrite:
        base = latest_results("rerank")
        name = "rerank"
        if not base:
            base = latest_results("baseline")
            name = "baseline"
        if base:
            print_lift(base, payload, name)
        else:
            print("(no prior results to compare against.)")
    elif args.hybrid and args.rerank:
        base = latest_results("rerank")
        name = "rerank"
        if not base:
            base = latest_results("baseline")
            name = "baseline"
        if base:
            print_lift(base, payload, name)
    elif args.hybrid:
        base = latest_results("baseline")
        if base:
            print_lift(base, payload, "baseline")
    elif args.rerank:
        base = latest_results("baseline")
        if base:
            print_lift(base, payload, "baseline")
    else:
        print("this is the number reranking / rewriting must beat. keep the file.")


if __name__ == "__main__":
    sys.exit(main())
