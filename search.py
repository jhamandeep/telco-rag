"""
search.py - ask the index a question in plain English.

Embeds the query with the SAME model as the corpus, cosine-searches telco_ran,
and prints the ranked matches. Handles BOTH payload shapes (troubleshooting
scenarios and spec chunks) and shows a snippet of what actually matched.

Run:  python search.py "your question here"
"""
import sys
from store import get_client
from embed import embed_one
import config


def _label(p: dict) -> str:
    if p.get("source") == "spec":
        return f"spec {p.get('spec','?')}  clause {p.get('clause','?')}  [{p.get('ctype','')}]"
    return f"{p.get('id','?')}  [{p.get('failure_mode','')}]"


def search(query: str, k: int = 5):
    c = get_client()
    qv = embed_one(query)
    hits = c.query_points(config.COLLECTION, query=qv, limit=k, with_payload=True).points
    print(f"\nQuery: {query}\n" + "-" * 72)
    for i, h in enumerate(hits, 1):
        p = h.payload or {}
        snippet = " ".join((p.get("text", "") or "").split())[:110]
        print(f"{i}. [{h.score:.3f}]  {p.get('source','?'):<15} {_label(p)}")
        print(f"     {snippet}...")
    return hits


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or \
        "UE keeps dropping during a fast handover and reconnects to the old cell"
    search(q)
