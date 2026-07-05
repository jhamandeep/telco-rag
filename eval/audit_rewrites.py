"""
audit_rewrites.py - inspect the rewrite strings saved in a results file and
flag formats that silently break the BM25 lane.

Motivation: qwen scored 0.734 vs gpt-5.5's 1.000 on the SAME pipeline. The
difference is rewrite quality - specifically formats the tokenizer can't split
('SCGFailure' -> one dead token). This tool proves it per query, using the
SAME tokenizer as bm25_tool, and reports each rewrite's lexical overlap with
its gold doc (overlap 0 = the br lane fetched nothing useful for that query).

Run:
    python eval\\audit_rewrites.py                       # newest rewrite run
    python eval\\audit_rewrites.py results_...161714.json  # a specific run
    python eval\\audit_rewrites.py --tier hard             # hard queries only
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm25_tool import tokenize          # the exact tokenizer the lane uses

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "eval" / "gold_set_v1.jsonl"
CAMEL = re.compile(r"[a-z][A-Z]")       # 'scgFailure' / 'SCGFailure' boundary


def newest_rewrite_run():
    files = sorted(glob.glob(str(ROOT / "eval" / "results_*rewrite*.json")))
    return files[-1] if files else None


def gold_tokens_by_qid():
    """Map qid -> set of content tokens across its gold docs (from the corpus)."""
    rel = {}
    for line in open(GOLD, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rel[r["qid"]] = set(r["relevant"])
    texts = {}
    for line in open(ROOT / "corpus" / "troubleshooting.jsonl", encoding="utf-8"):
        r = json.loads(line)
        texts[r["id"]] = r
    out = {}
    for qid, ids in rel.items():
        toks = set()
        for tid in ids:
            rec = texts.get(tid, {})
            blob = " ".join(str(rec.get(f, "")) for f in
                            ("title", "symptom", "text", "failure_mode",
                             "root_cause", "fix"))
            toks |= set(tokenize(blob))
        out[qid] = toks
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="results_*.json (default: newest rewrite run)")
    ap.add_argument("--tier", choices=["easy", "medium", "hard"], help="filter")
    a = ap.parse_args()

    path = a.file or newest_rewrite_run()
    if not path:
        raise SystemExit("no rewrite results file found")
    if not Path(path).is_absolute() and not Path(path).exists():
        path = str(ROOT / "eval" / path)
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    gen = d.get("config", {}).get("generator", "?")
    print(f"file: {Path(path).name}    generator: {gen}\n")

    gtok = gold_tokens_by_qid()
    rows = [r for r in d.get("per_query", []) if "rewrite" in r]
    if a.tier:
        rows = [r for r in rows if r.get("tier") == a.tier]
    if not rows:
        raise SystemExit("no per-query rewrite strings in this file "
                         "(re-run run_eval after the audit patch)")

    n_broken = n_zero = 0
    print(f"{'qid':<5}{'tier':<7}{'RR':>5}  flags  overlap  rewrite")
    print("-" * 78)
    for r in sorted(rows, key=lambda x: x["qid"]):
        rw = r["rewrite"]
        flags = []
        if CAMEL.search(rw):
            flags.append("CAMEL")
        if "," in rw:
            flags.append("COMMA")
        rw_tok = set(tokenize(rw))
        ov = len(rw_tok & gtok.get(r["qid"], set()))
        if ov == 0:
            flags.append("NOMATCH")
            n_zero += 1
        if flags:
            n_broken += 1
        flagstr = ",".join(flags) if flags else "ok"
        print(f"{r['qid']:<5}{r.get('tier',''):<7}{r.get('rr',0):>5.2f}  "
              f"{flagstr:<14}{ov:>4}   {rw[:60]}")

    print("-" * 78)
    print(f"{len(rows)} rewrites: {n_broken} flagged, {n_zero} with ZERO gold "
          f"overlap (br lane fetched nothing for those).")
    print("CAMEL/COMMA = tokenizer-breaking format; NOMATCH = no lexical bridge "
          "to the gold\n(dense lane may still carry it, but the lexical rescue "
          "is dead for that query).")


if __name__ == "__main__":
    main()
