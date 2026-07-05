"""
rrf.py - Reciprocal Rank Fusion (rung 7b): combine ranked lists from lanes
whose scores live on different scales (dense cosine ~0.7 vs BM25 sums ~15-80).

Scores never cross lanes; RANKS do. Each list votes 1/(k + rank) for each of
its docs; a doc's fused score is the sum of its votes; absence = no vote.

    RRF(d) = sum over lists containing d of  1 / (k + rank_in_list(d))

Why k = 60 (Cormack, Clarke & Buettcher, SIGIR 2009): with k=0, rank 1 scores
1.0 vs rank 2's 0.5 - one lane's top pick is a dictator. With k=60, rank 1
(0.0164) barely beats rank 2 (0.0161), so AGREEMENT across lanes outvotes any
single lane's favourite: ranks 15 & 20 in both lanes (1/75 + 1/80 = 0.0258)
beat a solo rank 1 (1/61 = 0.0164). The paper found k=60 empirically on TREC
and the field has never really re-tuned it.

Division of labour in our pipeline: RRF decides WHICH candidates deserve the
expensive cross-encoder (pool selection / recall); the cross-encoder decides
their final order (precision). RRF is also the cheap fused list the agent's
hybrid_search tool returns at rung 7d without paying for the cross-encoder.

Pure standard-library, like metrics.py: it fuses anything with an `.id`
attribute (Qdrant ScoredPoint and BM25Hit both qualify), so it is testable
with no services running.

Self-test (hand-traced, then asserted):  python eval\\rrf.py
"""
from collections import namedtuple

RRF_K = 60


def rrf_fuse(*ranked_lists, k=RRF_K):
    """Fuse ranked lists of hit objects (each hit has .id).
    Returns [(hit, fused_score)] sorted by score desc; ties broken by str(id)
    so runs are deterministic. The FIRST occurrence of a doc supplies the hit
    object kept (payloads are identical across lanes - same storage point)."""
    scores, keep = {}, {}
    for lst in ranked_lists:
        for rank, h in enumerate(lst, start=1):
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (k + rank)
            if h.id not in keep:
                keep[h.id] = h
    order = sorted(scores, key=lambda i: (-scores[i], str(i)))
    return [(keep[i], scores[i]) for i in order]


# --------------------------------------------------------------------------
# Self-test: the same discipline as metrics.py - trace by hand, then assert.
# --------------------------------------------------------------------------
def _selftest():
    H = namedtuple("H", "id")

    def ids(fused):
        return [h.id for h, _ in fused]

    print("worked examples (traced by hand, then asserted)\n" + "=" * 60)

    # A) agreement beats a solo star at k=60.
    #    lane1: X(1) Z(2) Y(3)     lane2: Z(1) W(2) Y(3)
    #    X = 1/61            = 0.016393
    #    Z = 1/62 + 1/61     = 0.032522
    #    Y = 1/63 + 1/63     = 0.031746
    #    W = 1/62            = 0.016129
    lane1 = [H("X"), H("Z"), H("Y")]
    lane2 = [H("Z"), H("W"), H("Y")]
    fused = rrf_fuse(lane1, lane2)
    print("A  k=60:", [(i, round(s, 6)) for i, s in
                       [(h.id, s) for h, s in fused]])
    assert ids(fused) == ["Z", "Y", "X", "W"]
    assert abs(fused[0][1] - (1 / 62 + 1 / 61)) < 1e-9
    assert abs(fused[2][1] - (1 / 61)) < 1e-9

    # B) small k = dictatorship. Solo rank 1 vs a pair at ranks 8 & 9.
    #    Crossover is near k ~ 6.5: below it the star wins, above it the pair.
    lane1 = [H("X")] + [H(f"f{i}") for i in range(6)] + [H("P")]   # P at rank 8
    lane2 = [H(f"g{i}") for i in range(8)] + [H("P")]              # P at rank 9
    lo = rrf_fuse(lane1, lane2, k=2)     # X = 1/3 = .333 > P = 1/10+1/11 = .191
    hi = rrf_fuse(lane1, lane2, k=60)    # X = 1/61 = .0164 < P = 1/68+1/69 = .0292
    print("B  k=2 :", "X above P ->", ids(lo).index("X") < ids(lo).index("P"))
    print("   k=60:", "P above X ->", ids(hi).index("P") < ids(hi).index("X"))
    assert ids(lo).index("X") < ids(lo).index("P")
    assert ids(hi).index("P") < ids(hi).index("X")

    # C) determinism: equal scores tie-break by str(id), every run.
    t = rrf_fuse([H("b"), H("a")], [H("a"), H("b")])   # both = 1/61 + 1/62
    print("C  tie :", ids(t))
    assert ids(t) == ["a", "b"]

    print("\nall asserts passed - rrf_fuse behaves as hand-traced.")


if __name__ == "__main__":
    _selftest()
