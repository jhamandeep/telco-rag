"""
validate_gold.py - sanity-check the gold set BEFORE any metric is trusted.

A metric computed on a broken answer key is worse than no metric. This script
enforces three hard rules (exit 1 on failure) and one advisory lint:

Hard rules
  1. Every line parses; qid values are unique; required fields are present.
  2. difficulty is easy | medium | hard; relevant is a non-empty list.
  3. Every relevant id resolves to a scenario in corpus/troubleshooting.jsonl.

Advisory lint (warn only)
  4. Contamination: containment C = |Q intersect D| / |Q| over content tokens,
     where Q = query tokens and D = tokens of the scenario text that was
     actually embedded (same compose() as ingest_troubleshooting.py).
     C near 1.0 means the query re-uses the document's own words - retrieval
     would then be graded on string overlap, not meaning. WARN at C >= 0.85.

Usage
  python eval\\validate_gold.py             summary + per-query table
  python eval\\validate_gold.py --show Q05  token-level overlap for one query
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "troubleshooting.jsonl"
GOLD = ROOT / "eval" / "gold_set_v1.jsonl"

TIERS = {"easy", "medium", "hard"}
WARN_CONTAINMENT = 0.85

# Function words only. Domain tokens (a3, t310, xn, rach...) are never dropped.
STOP = set("""
the a an and or of to in on for with is are was were be been being that this
it its as at by from do does did how what which when where who why i we you
they my our your their me us them so not no but if then than too very just
still yet ever never now here there into onto over under between within
without during after before while again back down up out off should would
can could may might must both same also only even all any some more most
less least own other per via keep keeps
""".split())


def compose(rec: dict) -> str:
    """Mirror ingest_troubleshooting.compose - lint against what was embedded."""
    return (
        f"{rec['title']}. "
        f"Symptoms: {rec['symptoms']} "
        f"Root cause: {rec['root_cause']} "
        f"Resolution: {rec['resolution']}"
    )


def tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP}


def containment(query: str, doc_text: str) -> tuple:
    q, d = tokens(query), tokens(doc_text)
    if not q:
        return 0.0, set(), q
    hit = q & d
    return len(hit) / len(q), hit, q


def main() -> int:
    show_qid = None
    if len(sys.argv) == 3 and sys.argv[1] == "--show":
        show_qid = sys.argv[2].upper()

    scenarios = {}
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            scenarios[rec["id"]] = compose(rec)

    errors, warnings = [], []
    rows, seen_qids = [], set()
    covered, tier_count = set(), {t: 0 for t in TIERS}
    multi = 0

    with open(GOLD, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                g = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {n}: bad JSON ({e})")
                continue

            qid = g.get("qid", f"line{n}")
            if qid in seen_qids:
                errors.append(f"{qid}: duplicate qid")
            seen_qids.add(qid)

            for field in ("query", "relevant", "difficulty"):
                if field not in g:
                    errors.append(f"{qid}: missing field '{field}'")

            tier = g.get("difficulty", "?")
            if tier not in TIERS:
                errors.append(f"{qid}: difficulty '{tier}' not in {sorted(TIERS)}")
            else:
                tier_count[tier] += 1

            rel = g.get("relevant", [])
            if not isinstance(rel, list) or not rel:
                errors.append(f"{qid}: 'relevant' must be a non-empty list")
                rel = []
            if len(rel) > 1:
                multi += 1

            worst = 0.0
            for rid in rel:
                if rid not in scenarios:
                    errors.append(f"{qid}: relevant id '{rid}' not in corpus")
                    continue
                covered.add(rid)
                c, hit, q = containment(g.get("query", ""), scenarios[rid])
                worst = max(worst, c)
                if show_qid == qid:
                    print(f"\n{qid} vs {rid}  C = {len(hit)}/{len(q)} = {c:.2f}")
                    print(f"  query tokens : {' '.join(sorted(q))}")
                    print(f"  shared       : {' '.join(sorted(hit))}")
                    print(f"  fresh        : {' '.join(sorted(q - hit))}")
            if worst >= WARN_CONTAINMENT:
                warnings.append(f"{qid}: containment {worst:.2f} >= {WARN_CONTAINMENT} - rephrase")

            rows.append((qid, tier, ",".join(rel), worst))

    if show_qid:
        return 0

    print(f"gold set : {GOLD.name}")
    print(f"queries  : {len(rows)}   "
          f"(easy {tier_count['easy']} / medium {tier_count['medium']} / hard {tier_count['hard']}, "
          f"{multi} multi-relevant)")
    print(f"coverage : {len(covered)}/{len(scenarios)} scenarios referenced")
    missing = sorted(set(scenarios) - covered)
    if missing:
        print(f"           uncovered: {', '.join(missing)}")
    cs = [r[3] for r in rows]
    print(f"containment (query words also found in its answer doc):")
    print(f"           min {min(cs):.2f}   mean {sum(cs)/len(cs):.2f}   max {max(cs):.2f}   warn at {WARN_CONTAINMENT}")
    print("-" * 64)
    for qid, tier, rel, c in rows:
        flag = "  WARN" if c >= WARN_CONTAINMENT else ""
        print(f"{qid}  {tier:<7} C={c:.2f}  -> {rel}{flag}")
    print("-" * 64)

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print(f"result   : {'FAIL' if errors else 'PASS'} "
          f"({len(errors)} errors, {len(warnings)} warnings)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
