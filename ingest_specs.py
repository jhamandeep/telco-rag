"""
ingest_specs.py - the spec lane: PDF -> extract -> chunk -> embed -> upsert.

Lands clause chunks into the SAME telco_ran collection as the troubleshooting
scenarios, tagged source="spec". Drop spec PDFs into data/ first.

Embeds in BATCHES (one Ollama call per EB chunks) - the difference between
minutes and hours on a big spec.

Run:  python ingest_specs.py
"""
import re
import glob
import uuid
from tqdm import tqdm
from qdrant_client.models import Distance, VectorParams, PointStruct
from store import get_client
from embed import embed_batch
from extract import extract_text
from chunker import chunk_document
import config

EB = 64            # embed-batch size (chunks per Ollama call)
UPSERT_EVERY = 256  # flush points to Qdrant in blocks


def infer_spec(filename: str) -> str:
    """ETSI 'ts_138331...' -> '38.331'; or a filename containing '38.331'."""
    m = re.search(r'1(\d{2})[ _]?(\d{3})', filename) or re.search(r'(\d{2})\.(\d{3})', filename)
    return f"{m.group(1)}.{m.group(2)}" if m else "unknown"


def ensure_collection(c):
    if not c.collection_exists(config.COLLECTION):
        c.create_collection(
            config.COLLECTION,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
        )


def main():
    c = get_client()
    ensure_collection(c)

    pdfs = sorted(glob.glob("data/*.pdf"))
    if not pdfs:
        print("no PDFs found in data/ - download at least one spec first.")
        return

    grand = 0
    for pdf in pdfs:
        spec = infer_spec(pdf)
        text = extract_text(pdf)
        chunks = chunk_document(text, spec=spec, mode=config.CHUNK_MODE)
        print(f"{pdf}: spec {spec}, {len(text):,} chars -> {len(chunks)} chunks (mode={config.CHUNK_MODE})")

        points = []
        for i in tqdm(range(0, len(chunks), EB), desc=f"embedding {spec}", unit="batch"):
            group = chunks[i:i + EB]
            vecs = embed_batch([ch.text for ch in group])
            for ch, vec in zip(group, vecs):
                pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"telcorag:spec:{ch.cid}"))
                points.append(PointStruct(id=pid, vector=vec, payload={**ch.payload(), "source": "spec"}))
            if len(points) >= UPSERT_EVERY:
                c.upsert(config.COLLECTION, points=points)
                points = []
        if points:
            c.upsert(config.COLLECTION, points=points)
        grand += len(chunks)

        for ch in chunks[:3]:
            print(f"    [{ch.clause:>8} | {ch.ctype:>11} | {ch.n_chars:4d}ch] {ch.text[:58].strip()}...")

    total = c.count(config.COLLECTION, exact=True).count
    print(f"\ningested {grand} spec chunks; collection '{config.COLLECTION}' now holds {total} points total")


if __name__ == "__main__":
    main()
