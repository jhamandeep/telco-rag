"""
check_corpus.py - is the collection we're searching the SAME one we froze the
baseline on? Run this BEFORE trusting any eval number.

A valid corpus is ~26,203 points: 16 troubleshooting + ~26,187 spec.
If spec == 0 (or total is tiny), the spec lane is missing - which makes the
retrieval task trivial (top-10 of ~16 almost always contains the gold) and
inflates every metric. That is a regression disguised as a win, and it makes
today's run NOT comparable to the frozen baseline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from store import get_client
from qdrant_client import models as m

c = get_client()
print("QDRANT_MODE :", config.QDRANT_MODE)
print("collection  :", config.COLLECTION)

try:
    total = c.count(config.COLLECTION, exact=True).count
except Exception as e:
    print("!! could not count - collection missing or client not connected:")
    print("  ", e)
    sys.exit(1)

print("total points:", total)
for src in ("troubleshooting", "spec"):
    n = c.count(
        config.COLLECTION,
        count_filter=m.Filter(must=[m.FieldCondition(
            key="source", match=m.MatchValue(value=src))]),
        exact=True,
    ).count
    print(f"  {src:<15}: {n}")

print("-" * 48)
if total < 1000:
    print("*** WARNING: far below ~26,203 - the spec lane looks ABSENT. ***")
    print("Today's metrics are measured on a tiny corpus and are NOT comparable")
    print("to the frozen baseline. Re-ingest specs before trusting any number:")
    print("    python ingest_specs.py")
    print("then re-confirm this shows ~26,203, and re-run the eval.")
else:
    print("corpus size looks right - safe to compare against the baseline.")
