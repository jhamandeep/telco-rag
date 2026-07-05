"""
compare_runs.py - tabulate every results_*.json for the paper, and trace a
single query across runs (the tool the FINDINGS backlog asked for).

Two uses:
  python eval\\compare_runs.py              # master table, chronological
  python eval\\compare_runs.py --qid Q05    # one query's fate across all runs

The per-query trace exists because aggregate deltas hide mechanisms: hard
R@10 = 0.750 has several decompositions, and only the qid trace says which
query paid. Pure stdlib; reads only the recorded JSON - never recomputes.
"""
import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_runs():
    runs = []
    for f in sorted(glob.glob(str(ROOT / "eval" / "results_*.json"))):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            d["_file"] = Path(f).name
            runs.append(d)
        except (json.JSONDecodeError, OSError) as e:
            print(f"!! skipping {Path(f).name}: {e}")
    runs.sort(key=lambda d: d.get("timestamp", ""))
    return runs


def _gated(d):
    pq = d.get("per_query", [])
    if not pq or "gate" not in pq[0]:
        return "-"
    return f"{sum(1 for r in pq if r.get('gate'))}/{len(pq)}"


def master_table(runs):
    hdr = (f"{'stamp':<16}{'run':<26}{'generator':<20}{'qmode':<8}{'pool':>5}{'flr':>5}{'gated':>7}"
           f"{'MRR':>7}{'R@5':>7}{'R@10':>7}{'hMRR':>7}{'hR@10':>7}")
    print(hdr + "\n" + "-" * len(hdr))
    for d in runs:
        o = d["summary"]["overall"]
        h = d["summary"]["by_tier"].get("hard", {})
        c = d.get("config", {})
        print(f"{d.get('timestamp',''):<16}{d.get('run',''):<26}"
              f"{str(c.get('generator') or '-'):<20}"
              f"{str(c.get('qdrant_mode') or '-'):<8}"
              f"{str(c.get('pool_k') or '-'):>5}"
              f"{str(c.get('ce_floor') if c.get('ce_floor') is not None else '-'):>5}"
              f"{_gated(d):>7}"
              f"{o['mrr']:>7.3f}{o['r@5']:>7.3f}{o['r@10']:>7.3f}"
              f"{h.get('mrr', 0):>7.3f}{h.get('r@10', 0):>7.3f}")


def qid_trace(runs, qid):
    hdr = (f"{'stamp':<16}{'run':<26}{'generator':<20}{'pool':>5}"
           f"{'RR':>7}{'R@5':>7}{'R@10':>7}")
    print(f"trace: {qid}\n" + hdr + "\n" + "-" * len(hdr))
    for d in runs:
        rec = next((r for r in d.get("per_query", [])
                    if r.get("qid") == qid), None)
        if rec is None:
            continue
        c = d.get("config", {})
        print(f"{d.get('timestamp',''):<16}{d.get('run',''):<26}"
              f"{str(c.get('generator') or '-'):<20}"
              f"{str(c.get('pool_k') or '-'):>5}"
              f"{rec['rr']:>7.3f}{rec['r@5']:>7.3f}{rec['r@10']:>7.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", type=str, help="trace one query id across all runs")
    a = ap.parse_args()
    runs = load_runs()
    if not runs:
        raise SystemExit("no results_*.json found in eval\\")
    if a.qid:
        qid_trace(runs, a.qid.upper())
    else:
        master_table(runs)
