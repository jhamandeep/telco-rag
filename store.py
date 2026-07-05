"""
store.py - returns a Qdrant client per config.

"local"  : embedded Qdrant inside THIS Python process, persisted to a folder.
           No Docker, no server, no daemon. Ideal for single-box dev.
           One process opens the storage at a time (fine for our sequential scripts).
"server" : talk to a running Qdrant (Docker/remote) over HTTP. Needed only if a
           later step wants a server-only feature or the web dashboard.

Flip with QDRANT_MODE in .env - no code changes anywhere else.
"""
from qdrant_client import QdrantClient
import config


def get_client() -> QdrantClient:
    if config.QDRANT_MODE == "local":
        return QdrantClient(path=config.QDRANT_PATH)   # embedded, on-disk
    return QdrantClient(url=config.QDRANT_URL)          # server (Docker/remote)
