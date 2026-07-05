"""
metrics.py - the scoring core of the eval harness. Pure standard-library:
no Qdrant, no Ollama, no numpy. It scores a *ranking* against a *gold set*,
so it does not care where the ranking came from - that decoupling is the
point (the same metrics grade dense retrieval today and the reranker later).

Vocabulary
  ranked   : list of point labels in rank order, rank 1 first.
             troubleshooting hits -> "TS-04"; spec hits -> "spec:38.331 5.3".
             The spec labels never equal a TS id, so they never count as gold.
  relevant : the set of gold labels for this query, e.g. {"TS-04"}.

Formulas (K is the cutoff; g = |gold in top-K|; G = |relevant|)
  Precision@K = g / K            fixed denominator K -> "of K slots, how many paid off"
  Recall@K    = g / G            denominator = gold that exists -> "of the answers, how many caught"
  RR          = 1 / rank1        rank of the FIRST gold hit; 0 if none in the list
  MRR         = mean(RR) over all queries

Run the self-test (no services needed):
  python eval\\metrics.py
"""
from statistics import mean

KS = (1, 3, 5, 10)  # cutoffs reported everywhere


def precision_at_k(ranked, relevant, k):
    """g / k. Denominator is k even if fewer than k gold exist - that is the
    Precision@K ceiling we discussed: 1 gold, k=5 -> best possible 0.20."""
    topk = ranked[:k]
    g = sum(1 for r in topk if r in relevant)
    return g / k


def recall_at_k(ranked, relevant, k):
    """g / G. If G > k you cannot reach 1.0 even when perfect (can't fit G
    answers in k slots). In this gold set max G = 3 and k>=5, so it's reachable."""
    G = len(relevant)
    if G == 0:
        return 0.0
    # distinct gold found (a set), so a duplicated gold label can never push
    # recall above 1.0. Identical to position-counting whenever the corpus is
    # deduped (our case), but correct even when it isn't.
    found = {r for r in ranked[:k] if r in relevant}
    return len(found) / G


def reciprocal_rank(ranked, relevant):
    """1 / (rank of first gold hit). 0.0 if no gold appears in the ranking."""
    for rank, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def score_query(ranked, relevant, ks=KS):
    """All metrics for one query -> flat dict: p@1..p@10, r@1..r@10, rr."""
    relevant = set(relevant)
    out = {}
    for k in ks:
        out[f"p@{k}"] = precision_at_k(ranked, relevant, k)
        out[f"r@{k}"] = recall_at_k(ranked, relevant, k)
    out["rr"] = reciprocal_rank(ranked, relevant)
    return out


def aggregate(records, ks=KS):
    """records: list of {qid, tier, **score_query(...)}.
    Returns overall means (MRR = mean rr) and the same, sliced per tier."""
    def means(rows):
        if not rows:
            return {}
        keys = [f"p@{k}" for k in ks] + [f"r@{k}" for k in ks] + ["rr"]
        m = {key: mean(r[key] for r in rows) for key in keys}
        m["mrr"] = m.pop("rr")
        m["n"] = len(rows)
        return m

    summary = {"overall": means(records), "by_tier": {}}
    for tier in ("easy", "medium", "hard"):
        rows = [r for r in records if r.get("tier") == tier]
        if rows:
            summary["by_tier"][tier] = means(rows)
    return summary


def format_table(summary, ks=KS, title="baseline"):
    """A compact terminal table. Headline row is P@5 / R@5 / MRR."""
    o = summary["overall"]
    lines = []
    lines.append(f"=== {title}  (n={o['n']} queries) ===")
    lines.append(f"HEADLINE   MRR {o['mrr']:.3f}   R@5 {o['r@5']:.3f}   P@5 {o['p@5']:.3f}")
    lines.append("-" * 60)
    hdr = "slice      " + "".join(f"P@{k:<5}" for k in ks) + "  " + "".join(f"R@{k:<5}" for k in ks) + "MRR"
    lines.append(hdr)
    def row(name, m):
        p = "".join(f"{m[f'p@{k}']:.3f} " for k in ks)
        r = "".join(f"{m[f'r@{k}']:.3f} " for k in ks)
        return f"{name:<10} {p} {r}{m['mrr']:.3f}"
    lines.append(row(f"all ({o['n']})", o))
    for tier, m in summary["by_tier"].items():
        lines.append(row(f"{tier} ({m['n']})", m))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Self-test: hand-traced examples. Run `python eval\metrics.py` - all asserts
# must pass and the worked numbers must match what we reasoned by hand.
# --------------------------------------------------------------------------
def _approx(a, b, tol=1e-3):
    return abs(a - b) < tol


def _selftest():
    print("worked examples (traced by hand, then asserted)\n" + "=" * 60)

    # A) single-gold query, gold at rank 1
    A = ["TS-04", "TS-05", "spec:38.331 5.3", "TS-02", "spec:38.133 9"]
    sA = score_query(A, {"TS-04"})
    print("A  ranked:", A)
    print(f"   gold={{TS-04}}  P@1={sA['p@1']:.3f} P@3={sA['p@3']:.3f} "
          f"P@5={sA['p@5']:.3f} R@5={sA['r@5']:.3f} RR={sA['rr']:.3f}")
    assert _approx(sA["p@1"], 1.0) and _approx(sA["p@3"], 1/3)
    assert _approx(sA["p@5"], 0.2) and _approx(sA["r@5"], 1.0)
    assert _approx(sA["rr"], 1.0)

    # B) same set, gold demoted to rank 3 -> P@5/R@5 unchanged, RR drops
    B = ["TS-05", "spec:38.331 5.3", "TS-04", "TS-02", "spec:38.133 9"]
    sB = score_query(B, {"TS-04"})
    print("B  ranked:", B)
    print(f"   gold={{TS-04}}  P@1={sB['p@1']:.3f} P@3={sB['p@3']:.3f} "
          f"P@5={sB['p@5']:.3f} R@5={sB['r@5']:.3f} RR={sB['rr']:.3f}")
    assert _approx(sB["p@1"], 0.0) and _approx(sB["p@5"], 0.2)
    assert _approx(sB["r@5"], 1.0) and _approx(sB["rr"], 1/3)
    assert _approx(sB["p@5"], sA["p@5"]) and _approx(sB["r@5"], sA["r@5"])  # order-blind
    assert not _approx(sB["rr"], sA["rr"])                                  # order-aware

    # C) multi-gold query (Q07-like): 3 correct answers
    C = ["TS-01", "spec:38.331 6", "TS-10", "spec:38.133 9", "TS-02"]
    sC = score_query(C, {"TS-01", "TS-02", "TS-10"})
    print("C  ranked:", C)
    print(f"   gold={{TS-01,TS-02,TS-10}}  P@3={sC['p@3']:.3f} P@5={sC['p@5']:.3f} "
          f"R@3={sC['r@3']:.3f} R@5={sC['r@5']:.3f} RR={sC['rr']:.3f}")
    assert _approx(sC["p@3"], 2/3) and _approx(sC["p@5"], 0.6)   # 2 of top3, 3 of top5
    assert _approx(sC["r@3"], 2/3) and _approx(sC["r@5"], 1.0)   # caught 2/3 then 3/3
    assert _approx(sC["rr"], 1.0)                                # TS-01 at rank 1

    # aggregate demo -> MRR = mean(1.0, 0.333, 1.0) = 0.778
    recs = [
        {"qid": "A", "tier": "easy", **sA},
        {"qid": "B", "tier": "hard", **sB},
        {"qid": "C", "tier": "medium", **sC},
    ]
    summ = aggregate(recs)
    print("\n" + format_table(summ, title="self-test"))
    assert _approx(summ["overall"]["mrr"], (1.0 + 1/3 + 1.0) / 3)

    print("\nall asserts passed - the math is correct.")


if __name__ == "__main__":
    _selftest()
