"""
run_agent.py - evaluate the ReAct agent (agent.py) on gold_set_v1, on the SAME
22 queries and the SAME metrics as every static run, so the agentic-vs-static
delta is a clean comparison. Records COST per query (tool calls, LLM calls,
stop step) next to the retrieval metrics, because the agent's headline is
quality PER tool-call, not quality alone.

Wires the real, already-tested tools into the agent:
  dense   -> Qdrant query_points via embed_one (store/embed)
  bm25    -> bm25_tool.ensure_index(...).search (fingerprint-guarded)
  rewrite -> rewrite.rewrite_query (REWRITE_BACKEND selects local/openai)
  rerank  -> rerank.rerank (bge-reranker-base, gated inside the agent)

Run (Qdrant server + Ollama up):
    python eval\\run_agent.py --show          # per-query trace + cost
    python eval\\run_agent.py --budget 2      # cap escalation at 2 steps
    $env:REWRITE_BACKEND="local"; python eval\\run_agent.py   # qwen rewrites

Compares against the strongest static run automatically (the rewrite+hybrid+
gate result) so the printout answers: did adaptivity match that quality, and
how many tool calls did it save vs the static "always 6 ops" pipeline?
"""
import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from store import get_client
from embed import embed_one
import config
import metrics
from bm25_tool import ensure_index
from rerank import rerank
from rewrite import rewrite_query, active_generator
from agent import Agent

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "eval" / "gold_set_v1.jsonl"


def load_gold():
    rows = []
    with open(GOLD, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_tools(client, bm25_index):
    """Adapt the real tools to the agent's uniform tool signature."""
    def dense(query, k):
        return client.query_points(config.COLLECTION, query=embed_one(query),
                                    limit=k, with_payload=True).points

    def bm25(query, k):
        return bm25_index.search(query, k=k)

    return {"dense": dense, "bm25": bm25,
            "rewrite": rewrite_query, "rerank": rerank}


def latest_static():
    """The strongest static baseline: newest hybrid+rewrite run WITH a gate."""
    best = None
    for f in sorted(glob.glob(str(ROOT / "eval" / "results_*rewrite*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if d.get("config", {}).get("ce_floor"):        # gated runs only
            best = d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lane-k", type=int, default=10, dest="lane_k")
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--ce-floor", type=float, default=0.5, dest="ce_floor")
    ap.add_argument("--show", action="store_true", help="per-query trace")
    args = ap.parse_args()

    gold = load_gold()
    client = get_client()
    bm25_index = ensure_index(client)
    tools = build_tools(client, bm25_index)
    agent = Agent(tools, k=args.k, lane_k=args.lane_k, budget=args.budget,
                  ce_floor=args.ce_floor)
    gen = active_generator()

    records, total_tools, total_llm = [], 0, 0
    step_hist = {1: 0, 2: 0, 3: 0}
    for g in gold:
        tr = agent.run(g["qid"], g["query"])
        s = metrics.score_query(tr.final_ranked, g["relevant"])
        stop_step = tr.steps[-1].n if tr.steps else 0
        rec = {"qid": g["qid"], "tier": g["difficulty"],
               "tool_calls": tr.tool_calls, "llm_calls": tr.llm_calls,
               "stop_step": stop_step, **s}
        records.append(rec)
        total_tools += tr.tool_calls
        total_llm += tr.llm_calls
        step_hist[stop_step] = step_hist.get(stop_step, 0) + 1

        if args.show:
            hit = "OK " if s["rr"] > 0 else "MISS"
            print(f"\n{g['qid']} [{g['difficulty']}] {hit} "
                  f"tools={tr.tool_calls} llm={tr.llm_calls} RR={s['rr']:.2f} "
                  f"R@10={s['r@10']:.2f}")
            for st in tr.steps:
                print(f"    step{st.n} {st.action:<16} top={st.top_label:<16}"
                      f" suff={st.sufficient}")
            print(f"    stop: {tr.stop_reason}")

    summary = metrics.aggregate(records)
    print("\n" + metrics.format_table(summary, title=f"AGENT (ReAct, {gen})"))

    # cost report - the agent's real headline
    n = len(records)
    static_ops = 6 * n     # static pipeline: 4 lanes + rewrite + rerank per query
    print(f"\n=== cost ({n} queries) ===")
    print(f"agent tool calls : {total_tools}  ({total_tools/n:.1f}/query)")
    print(f"agent LLM calls  : {total_llm}  ({total_llm/n:.2f}/query) "
          f"<- static pays {n} (one rewrite every query)")
    print(f"stop-step histogram: step1={step_hist.get(1,0)} "
          f"step2={step_hist.get(2,0)} step3={step_hist.get(3,0)}")
    print(f"LLM calls saved vs static: {n - total_llm}/{n} queries never "
          f"needed the rewrite.")

    static = latest_static()
    if static:
        so = static["summary"]["overall"]
        ao = summary["overall"]
        sg = static.get("config", {}).get("generator", "?")
        print(f"\n=== agent vs strongest static ({static.get('run','?')}, {sg}) ===")
        for m in ("mrr", "r@5", "r@10"):
            print(f"  {m.upper():<5} static {so[m]:.3f} -> agent {ao[m]:.3f} "
                  f"({ao[m]-so[m]:+.3f})")
        print("  read: matching quality at fewer LLM calls is the agentic win; "
              "a small quality dip is the cost of adaptivity (analyse in paper).")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"run": "agent_react", "timestamp": stamp,
               "config": {"generator": gen, "budget": args.budget,
                          "ce_floor": args.ce_floor, "k": args.k,
                          "lane_k": args.lane_k,
                          "qdrant_mode": config.QDRANT_MODE,
                          "cutoffs": list(metrics.KS),
                          "total_tool_calls": total_tools,
                          "total_llm_calls": total_llm},
               "summary": summary, "per_query": records}
    out = ROOT / "eval" / f"results_agent_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out.name}")


if __name__ == "__main__":
    main()
