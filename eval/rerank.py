"""
rerank.py - cross-encoder reranking of the dense candidate list.

Two-stage retrieval:
  stage 1 (bi-encoder, already built): embed query + passages SEPARATELY,
          compare by cosine. Fast enough to scan all 26k points, but coarse -
          the passage was encoded without ever seeing the query.
  stage 2 (cross-encoder, here): feed the PAIR [query, passage] through the
          model together so every query token attends to every passage token.
          Far sharper, but no precomputed vector - so we run it only on the
          top-K the bi-encoder already fetched.

The model returns ONE relevance logit per pair (higher = more relevant). We
sort the SAME candidate set by that score. Because it is the same set merely
reordered, R@K at the full depth cannot change - only the order (R@5, MRR) can.

First run downloads BAAI/bge-reranker-base (~1.1 GB) from HuggingFace; cached
after that. Uses the GPU if torch sees CUDA, else CPU (fine for eval-size batches).
"""
import config

_model = None


def get_reranker():
    """Lazy singleton - load the cross-encoder once, not per query."""
    global _model
    if _model is None:
        import torch
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = CrossEncoder(config.RERANK_MODEL, device=device)
    return _model


def _reorder(hits, scores):
    """Pure, mockable core: pair each hit with its score, sort desc, return
    list of (hit, score). Kept separate from the model so it is unit-testable
    without torch."""
    paired = list(zip(hits, (float(s) for s in scores)))
    paired.sort(key=lambda t: t[1], reverse=True)
    return paired


def rerank(query, hits):
    """Reorder dense hits with the cross-encoder.
    Returns list of (hit, rerank_score) sorted by score descending.
    Each hit must carry payload['text'] - the passage the model scores."""
    if not hits:
        return []
    model = get_reranker()
    pairs = [(query, (h.payload or {}).get("text", "")) for h in hits]
    scores = model.predict(pairs)
    return _reorder(hits, scores)
