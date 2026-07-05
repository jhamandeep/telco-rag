"""
autopsy.py - rebuild the hybrid+rewrite pipeline for ONE gold query and show
the judge's FULL verdict: every lane's view of the gold, the fused pool, and
the cross-encoder's score for every candidate - gold marked.

The aggregate metrics said WHAT (Q05 = 0.000 in every legitimate config);
this shows WHY (e.g. "TS-01 entered the pool at br rank 1, and the CE scored
it 0.0004, rank 27 of 40"). That number is the paper's evidence that the
residual failure is JUDGE-bound, not retrieval-bound.

Run:
    python eval\\autopsy.py Q05                  # 4 lanes, full-union pool
    python eval\\autopsy.py Q05 --no-rewrite     # 2 raw lanes only
    python eval\\autopsy.py Q13                  # inspect the fusion-tax case
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from store import get_client
from embed import embed_one
import config
from bm25_tool import ensure_index
from rrf import rrf_fuse
from rerank import rerank

GOLD = Path(__file__).resolve().parents[1] / "eval" / "gold_set_v1.jsonl"


def load_gold_row(qid):
    with open(GOLD, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["qid"] == qid:
                    return r
    raise SystemExit(f"{qid} not found in {GOLD.name}")


def label(p):
    return p["id"] if p.get("id") else f"spec:{p.get('spec','?')} {p.get('clause','?')}"


def lane_view(name, hits, gold):
    """Print one lane's top hits and where (if anywhere) the gold sits."""
    pos = {label(h.payload or {}): i for i, h in enumerate(hits, 1)}
    marks = [f"{g}@{pos[g]}" for g in gold if g in pos]
    print(f"  {name:<3} top{len(hits)}: gold {'-> ' + ', '.join(marks) if marks else 'ABSENT'}"
          f"   (head: {', '.join(label(h.payload or {}) for h in hits[:3])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qid")
    ap.add_argument("--k", type=int, default=10, help="depth per lane")
    ap.add_argument("--no-rewrite", action="store_true",
                    help="raw 2-lane autopsy (dense + bm25 on the original)")
    args = ap.parse_args()

    g = load_gold_row(args.qid.upper())
    gold = set(g["relevant"])
    client = get_client()
    bm25 = ensure_index(client)

    def dense(q):
        return client.query_points(config.COLLECTION, query=embed_one(q),
                                   limit=args.k, with_payload=True).points

    print(f"{g['qid']} [{g['difficulty']}]  gold={sorted(gold)}")
    print(f"  {g['query']}")

    lanes = {"do": dense(g["query"]), "bo": bm25.search(g["query"], k=args.k)}
    if not args.no_rewrite:
        from rewrite import rewrite_query, active_generator, PROMPT_TAG
        rw = rewrite_query(g["query"])
        print(f"  rewrite ({active_generator()}, prompt {PROMPT_TAG}):\n    {rw}")
        lanes["dr"] = dense(rw)
        lanes["br"] = bm25.search(rw, k=args.k)

    print("\n[1] each lane's own view of the gold:")
    for n, lst in lanes.items():
        lane_view(n, lst, gold)

    fused = rrf_fuse(*lanes.values())
    lane_ids = {n: {h.id for h in lst} for n, lst in lanes.items()}
    pool = [h for h, _ in fused]            # FULL union - nothing truncated
    rrf_rank = {h.id: i for i, (h, _) in enumerate(fused, 1)}

    print(f"\n[2] judge's verdict on the full union ({len(pool)} candidates):")
    print(f"  {'CE#':>4} {'CE score':>9} {'RRF#':>5} {'lanes':<12} label")
    for i, (h, s) in enumerate(rerank(g["query"], pool), 1):
        lbl = label(h.payload or {})
        tags = "+".join(n for n in ("do", "dr", "bo", "br") if h.id in lane_ids.get(n, ()))
        mark = "*" if lbl in gold else " "
        head = i <= 10
        if head or lbl in gold:             # show the kept-10 plus every gold
            print(f" {mark}{i:>4} {s:>9.4f} {rrf_rank[h.id]:>5} {tags:<12} {lbl}"
                  f"{'' if head else '   <- BELOW THE CUT'}")

    found = [lbl for lbl in gold if any(label(h.payload or {}) == lbl for h in pool)]
    print(f"\n[3] verdict: gold in pool: {found or 'NONE'}; "
          f"kept-10 = the CE's top 10 above. If a gold row says BELOW THE CUT,"
          f"\n    retrieval delivered it and the judge discarded it - judge-bound.")


if __name__ == "__main__":
    main()
