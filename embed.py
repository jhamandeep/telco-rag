"""
embed.py - the ONE place text becomes a vector.

Everything (corpus AND queries) goes through this, so every vector lives in the
same nomic-embed-text space - the rule that makes cosine comparisons meaningful.
"""
import requests
import config


def embed_one(text: str) -> list:
    r = requests.post(
        f"{config.OLLAMA_URL}/api/embeddings",
        json={"model": config.EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    d = r.json()
    vec = d.get("embedding") or d.get("embeddings", [[]])[0]
    if len(vec) != config.EMBED_DIM:
        raise ValueError(f"embed dim {len(vec)} != expected {config.EMBED_DIM}")
    return vec


def embed_batch(texts: list) -> list:
    """Embed MANY texts in one call via Ollama's /api/embed - the GPU processes
    the whole batch together, so this is dramatically faster than one request
    per text on a large corpus. Falls back to one-at-a-time if /api/embed is
    unavailable (older Ollama)."""
    try:
        r = requests.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": config.EMBED_MODEL, "input": texts},
            timeout=300,
        )
        r.raise_for_status()
        embs = r.json().get("embeddings")
        if embs and len(embs) == len(texts) and len(embs[0]) == config.EMBED_DIM:
            return embs
    except Exception:
        pass
    return [embed_one(t) for t in texts]  # safe fallback
