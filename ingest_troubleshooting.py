"""
ingest_troubleshooting.py - the "atomic" ingest lane.

Each JSONL scenario is already one self-contained unit, so there is no PDF
extraction and no chunking here: one scenario -> one embedded point.

Run:  python ingest_troubleshooting.py
Re-run safe: point IDs are a deterministic UUID of the scenario id, so a second
run overwrites rather than duplicating.
"""
import json
import uuid
from qdrant_client.models import Distance, VectorParams, PointStruct
from store import get_client
from embed import embed_one
import config

CORPUS = "corpus/troubleshooting.jsonl"


def compose(rec: dict) -> str:
    """The text that actually gets embedded - the whole scenario in one blob."""
    return (
        f"{rec['title']}. "
        f"Symptoms: {rec['symptoms']} "
        f"Root cause: {rec['root_cause']} "
        f"Resolution: {rec['resolution']}"
    )


def ensure_collection(c) -> None:
    if not c.collection_exists(config.COLLECTION):
        c.create_collection(
            config.COLLECTION,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"created collection '{config.COLLECTION}' ({config.EMBED_DIM}-D, cosine)")


def main() -> None:
    c = get_client()
    ensure_collection(c)

    points = []
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = compose(rec)
            vec = embed_one(text)
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"telcorag:troubleshooting:{rec['id']}"))
            payload = {**rec, "text": text, "source": "troubleshooting"}
            points.append(PointStruct(id=pid, vector=vec, payload=payload))

    c.upsert(config.COLLECTION, points=points)
    total = c.count(config.COLLECTION, exact=True).count
    print(f"upserted {len(points)} troubleshooting scenarios")
    print(f"collection '{config.COLLECTION}' now holds {total} points")


if __name__ == "__main__":
    main()
