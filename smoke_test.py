"""
smoke_test.py - foundation check for the single-box TelcoRAG.

Proves two things actually talk on THIS machine, before we build on top:
  1. Ollama embeds  (nomic-embed-text -> 768-dim vector).
  2. Qdrant stores that vector and returns it  (round-trip).

Prerequisites:
  ollama pull nomic-embed-text
  (Qdrant runs EMBEDDED by default - no Docker needed.)
"""
import sys
import requests
from qdrant_client.models import Distance, VectorParams, PointStruct
from store import get_client
import config

PROBE = "On radio link failure the UE starts T310 after N310 out-of-sync indications."


def check_embed() -> list:
    r = requests.post(
        f"{config.OLLAMA_URL}/api/embeddings",
        json={"model": config.EMBED_MODEL, "prompt": PROBE},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    vec = data.get("embedding") or data.get("embeddings", [[]])[0]
    assert len(vec) == config.EMBED_DIM, f"got dim {len(vec)}, expected {config.EMBED_DIM}"
    print(f"[OK]  Ollama embed       -> {len(vec)}-dim vector via '{config.EMBED_MODEL}'")
    return vec


def check_qdrant(vec: list) -> None:
    c = get_client()
    name = "smoke_test"
    if c.collection_exists(name):
        c.delete_collection(name)
    c.create_collection(
        name,
        vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
    )
    c.upsert(name, points=[PointStruct(id=1, vector=vec, payload={"note": "hello telco"})])
    res = c.query_points(name, query=vec, limit=1).points
    assert res and res[0].id == 1, "round-trip failed"
    print(f"[OK]  Qdrant round-trip   -> score={res[0].score:.4f}, payload={res[0].payload}")
    print(f"       (vector store mode: {config.QDRANT_MODE})")
    c.delete_collection(name)


if __name__ == "__main__":
    try:
        v = check_embed()
        check_qdrant(v)
        print("\nFoundation OK - embedder + vector store verified on this box.")
    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
