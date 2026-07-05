"""
bm25_tool.py - lexical (BM25) retrieval over the SAME frozen corpus (rung 7a).

Why this exists: Q05 is embedder-bound. Even gpt-5.5's clean rewrite never
placed TS-01 in the fetched pool, because nomic-embed-text will not put the
query vector near that doc. Dense retrieval matches MEANING; BM25 matches
EXACT TERMS weighted by rarity. They fail on different queries - which is
exactly what makes BM25 worth adding as a second lane (and, at rung 7d, a
tool the agent can CHOOSE).

One honest caveat, stated up front: BM25 can only score terms the query
CONTAINS. Q05's containment against its gold is C = 0.00 - the raw query and
TS-01 share zero content tokens - so raw-query BM25 scores TS-01 exactly 0.
The lexical lane cracks Q05 only when it is fed the REWRITTEN query (whose
spec terms overlap TS-01 heavily). --probe-q05 demonstrates both halves.

Design decisions (each one earns its place):
  * Index is built by SCROLLING Qdrant, not by re-parsing the PDFs. One frozen
    corpus, two views of it - the corpus-integrity lesson, encoded.
  * The cache carries a fingerprint {collection, point count, tokenizer tag}.
    On load it is checked against the LIVE point count; a mismatch forces a
    rebuild. A stale lexical index would be the corpus-wipe bug all over again.
  * Hits mimic Qdrant's ScoredPoint (.id / .score / .payload), with .id being
    the SAME storage point id - so merge_hits(), hit_label() and rerank()
    consume them unchanged at rung 7b (fusion) with zero refactoring.
  * Zero-score docs are never returned. "Matched nothing" is not a hit.

Tokenizer (tag v1 - changing it invalidates the cache, by design):
  lowercase, split on runs of [a-z0-9], drop a minimal stopword list.
  "A3 offset" -> a3, offset      "time-to-trigger" -> time, trigger
  "38.331"    -> 38, 331         "5G" -> 5g
  No stemming (rank_bm25 does none): calls != call. IDF absorbs most of the
  damage; the dense lane covers the rest - that asymmetry IS the hybrid case.

Dependency: rank_bm25 (pure-Python Okapi BM25, builds 26k docs in seconds,
no server, no re-ingest). The alternative - sparse vectors inside Qdrant -
would mean touching the frozen collection; a read-only side index does not.

Run (Qdrant server up):
    python eval\\bm25_tool.py --build            # scroll corpus, build, cache
    python eval\\bm25_tool.py --stats            # N, avgdl, IDF of probe terms
    python eval\\bm25_tool.py --query "handover too late A3 offset"
    python eval\\bm25_tool.py --probe-q05        # the rung-7a hypothesis test
    python eval\\bm25_tool.py --probe-q05 --with-rewrite   # + live qwen rewrite
"""
import argparse
import json
import math
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "bm25_cache.pkl"
GOLD = ROOT / "eval" / "gold_set_v1.jsonl"

TOKENIZER_TAG = "v1"
_WORD = re.compile(r"[a-z0-9]+")
STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "we", "you", "i",
    "do", "does", "did", "how", "what", "when", "where", "why", "not",
    "but", "if", "then", "than", "so", "can", "into", "while", "which",
}


def tokenize(text: str) -> list[str]:
    """Query and corpus MUST pass through the same tokenizer - the lexical
    twin of 'same embedder for corpus and query'."""
    return [t for t in _WORD.findall((text or "").lower()) if t not in STOP]


@dataclass
class BM25Hit:
    """Duck-typed to Qdrant's ScoredPoint: .id / .score / .payload.
    id is the SAME storage point id, so cross-lane dedup by id just works."""
    id: str
    score: float
    payload: dict = field(default_factory=dict)


class BM25Index:
    def __init__(self, ids, payloads, tokenized, fingerprint):
        from rank_bm25 import BM25Okapi   # import here: cheap CLI paths skip it
        self.ids = ids                    # position -> qdrant point id
        self.payloads = payloads          # position -> payload dict
        self.tokenized = tokenized        # position -> token list
        self.fingerprint = fingerprint
        self.bm25 = BM25Okapi(tokenized)  # k1=1.5, b=0.75 defaults

    # ---- search ------------------------------------------------------------
    def search(self, query: str, k: int = 10) -> list[BM25Hit]:
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)   # one score per corpus doc
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in order[:k]:
            if scores[i] <= 0.0:
                break                          # zero overlap is not a hit
            hits.append(BM25Hit(self.ids[i], float(scores[i]), self.payloads[i]))
        return hits

    # ---- worked-number helpers (used by --stats) ----------------------------
    def df(self, term: str) -> int:
        return sum(1 for doc in self.tokenized if term in doc)

    def idf_okapi(self, term: str) -> float:
        """The exact IDF rank_bm25's BM25Okapi uses: ln((N-n+0.5)/(n+0.5)).
        Negative values (term in >half the docs) get floored internally."""
        N, n = len(self.ids), self.df(term)
        if n == 0:
            return 0.0
        return math.log((N - n + 0.5) / (n + 0.5))

    def avgdl(self) -> float:
        return sum(len(d) for d in self.tokenized) / max(1, len(self.tokenized))

    def find_by_label(self, label: str):
        """Payload text for a semantic label like TS-01 (for the probe)."""
        for p in self.payloads:
            if p.get("id") == label:
                return p
        return None


# ---- build / cache ----------------------------------------------------------
def _scroll_corpus(client):
    """Pull every point's payload (no vectors) out of the frozen collection."""
    ids, payloads, offset = [], [], None
    while True:
        points, offset = client.scroll(
            config.COLLECTION, limit=1024,
            with_payload=True, with_vectors=False, offset=offset)
        for pt in points:
            ids.append(pt.id)
            payloads.append(pt.payload or {})
        if offset is None:
            return ids, payloads


def _live_count(client) -> int:
    return client.count(config.COLLECTION, exact=True).count


def build_index(client=None, quiet=False) -> BM25Index:
    if client is None:
        from store import get_client
        client = get_client()
    total = _live_count(client)
    if not quiet:
        print(f"scrolling {total} points from '{config.COLLECTION}' "
              f"(QDRANT_MODE={config.QDRANT_MODE}) ...")
    if total < 1000:
        print("*** WARNING: corpus far below ~26,203 - spec lane looks ABSENT."
              "\n*** A BM25 index over a wiped corpus poisons every comparison."
              "\n*** Run eval\\check_corpus.py / bootstrap.py first.")
    ids, payloads = _scroll_corpus(client)
    tokenized = [tokenize(p.get("text", "")) for p in payloads]
    fp = {"collection": config.COLLECTION, "points": len(ids),
          "tokenizer": TOKENIZER_TAG}
    idx = BM25Index(ids, payloads, tokenized, fp)
    CACHE.write_bytes(pickle.dumps(
        {"fingerprint": fp, "ids": ids, "payloads": payloads,
         "tokenized": tokenized}))
    if not quiet:
        print(f"built BM25 over {len(ids)} docs "
              f"(avgdl {idx.avgdl():.1f} tokens) -> cached {CACHE.name}")
    return idx


def ensure_index(client=None, force=False, quiet=True) -> BM25Index:
    """Load the cached index if its fingerprint matches the LIVE collection;
    otherwise (or on --force) rebuild. Callers at rung 7b use this one call."""
    if client is None:
        from store import get_client
        client = get_client()
    if not force and CACHE.exists():
        blob = pickle.loads(CACHE.read_bytes())
        fp = blob.get("fingerprint", {})
        live = _live_count(client)
        if (fp.get("collection") == config.COLLECTION
                and fp.get("tokenizer") == TOKENIZER_TAG
                and fp.get("points") == live):
            return BM25Index(blob["ids"], blob["payloads"],
                             blob["tokenized"], fp)
        print(f"cache stale (cached {fp.get('points')} pts, live {live}) "
              f"-> rebuilding")
    return build_index(client, quiet=quiet)


# ---- CLI ---------------------------------------------------------------------
def _label(p: dict) -> str:                       # same shape run_eval prints
    if p.get("id"):
        return p["id"]
    return f"spec:{p.get('spec', '?')} {p.get('clause', '?')}"


def _print_hits(hits, gold=None):
    gold = set(gold or [])
    if not hits:
        print("   (no doc matched any query term - score 0 everywhere)")
    for i, h in enumerate(hits, 1):
        mark = "*" if _label(h.payload) in gold else " "
        snippet = " ".join((h.payload.get("text", "") or "").split())[:88]
        print(f"  {mark}{i:>2}. [{h.score:7.3f}] {_label(h.payload):<22} {snippet}")


def _load_gold_row(qid: str) -> dict:
    with open(GOLD, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["qid"] == qid:
                    return r
    raise SystemExit(f"{qid} not found in {GOLD.name}")


def probe_q05(idx: BM25Index, with_rewrite: bool):
    g = _load_gold_row("Q05")
    gold_label = g["relevant"][0]
    gold_doc = idx.find_by_label(gold_label)
    if gold_doc is None:
        raise SystemExit(f"{gold_label} not in the index - corpus incomplete?")

    print(f"\nQ05 [{g['difficulty']}]  gold={g['relevant']}")
    print(f"  {g['query']}")

    q_toks, d_toks = set(tokenize(g["query"])), set(tokenize(gold_doc["text"]))
    overlap = sorted(q_toks & d_toks)
    print(f"\n[1] shared content tokens with {gold_label}: "
          f"{overlap if overlap else 'NONE'}")
    print("    -> every query term contributes 0 to this doc; its BM25 score"
          "\n       is exactly 0. No k, no reranker can surface a 0-score doc.")

    print("\n[2] BM25 on the RAW query (expect: gold ABSENT):")
    _print_hits(idx.search(g["query"], k=10), g["relevant"])

    spec_probe = ("too-late handover A3 offset time-to-trigger "
                  "radio link failure measurement report")
    print(f"\n[3] BM25 on spec-vocabulary terms (what a rewrite emits):"
          f"\n    '{spec_probe}'")
    _print_hits(idx.search(spec_probe, k=10), g["relevant"])

    if with_rewrite:
        from rewrite import rewrite_query, active_generator
        rw = rewrite_query(g["query"])
        print(f"\n[4] BM25 on a LIVE rewrite ({active_generator()}):"
              f"\n    '{rw}'")
        _print_hits(idx.search(rw, k=10), g["relevant"])
    print("\nverdict: the lexical lane needs the rewrite's vocabulary to see"
          "\nTS-01 - rewrite->BM25 is the chain, and choosing it is the agent's job.")


def show_stats(idx: BM25Index):
    N, avgdl = len(idx.ids), idx.avgdl()
    print(f"docs N = {N}    avgdl = {avgdl:.1f} tokens    "
          f"(k1=1.5, b=0.75)")
    print(f"{'term':<14}{'df(n)':>8}{'IDF=ln((N-n+.5)/(n+.5))':>26}")
    for t in ("handover", "a3", "offset", "trigger", "t310", "rlf",
              "measurement", "gap", "expressway", "cell"):
        print(f"{t:<14}{idx.df(t):>8}{idx.idf_okapi(t):>26.3f}")
    print("rarer term -> larger IDF -> each occurrence worth more.")


def main():
    ap = argparse.ArgumentParser(description="BM25 lexical lane (rung 7a)")
    ap.add_argument("--build", action="store_true", help="scroll corpus, build, cache")
    ap.add_argument("--force", action="store_true", help="rebuild even if cache is fresh")
    ap.add_argument("--stats", action="store_true", help="N, avgdl, IDF of probe terms")
    ap.add_argument("--query", type=str, help="run one BM25 search")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--probe-q05", action="store_true",
                    help="raw query vs spec-term query against the Q05 gold")
    ap.add_argument("--with-rewrite", action="store_true",
                    help="probe-q05 also BM25s a live LLM rewrite")
    args = ap.parse_args()

    if args.build or args.force:
        idx = build_index() if args.force else ensure_index(force=args.force,
                                                            quiet=False)
        print(f"fingerprint: {idx.fingerprint}")
    else:
        idx = ensure_index(quiet=False)

    if args.stats:
        show_stats(idx)
    if args.query:
        print(f"\nBM25 query: {args.query}")
        _print_hits(idx.search(args.query, k=args.k))
    if args.probe_q05:
        probe_q05(idx, with_rewrite=args.with_rewrite)
    if not any((args.build, args.force, args.stats, args.query, args.probe_q05)):
        ap.print_help()


if __name__ == "__main__":
    sys.exit(main())
