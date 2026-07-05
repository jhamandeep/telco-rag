"""
agent.py - ReAct agentic retriever (rung 7d).

Everything before this rung was a FIXED pipeline: it paid for all four lanes +
rewrite + rerank on every query, then gated the reranker post-hoc. An agent
instead DECIDES, per query, what to do next based on what it has already seen.
The ladder earned the thesis: the right action depends on the query, and a
fixed pipeline cannot exploit that.

  * easy queries      -> dense alone solves them (that tier = 1.000 since rerank)
  * exact-term queries-> BM25 catches what dense dilutes (Q14)
  * Q05               -> ONLY rewrite->BM25 places the gold; nothing else does

So the agent's win is NOT beating the 1.000 static ceiling. It is MATCHING
that quality at lower COST - escalating from cheap to expensive only when a
cheap action proves insufficient. The headline metric becomes quality PER
tool-call, which is the actual argument for agentic RAG.

ReAct loop (Thought -> Action -> Observation -> Reflect), step budget 3:
  step 1  ACTION dense_search            (cheapest; always first)
          REFLECT: sufficient? -> STOP
  step 2  ACTION bm25_search + RRF-fuse  (still cheap, no LLM)
          REFLECT -> STOP
  step 3  ACTION rewrite_query -> refetch BOTH lanes -> RRF-fuse   (LLM: last)
          finalize

REFLECTION CRITIC - the load-bearing decision. It must NOT read the cross-
encoder score: rungs 5/7c proved the CE collapses (scores ~0.000) on exactly
the hard queries where reflection matters. Instead it judges STRUCTURALLY,
using the signal the autopsies exposed - golds arrive as troubleshooting docs
that MULTIPLE lanes agree on (do+bo), junk arrives single-tagged:
    sufficient  <=>  top fused hit is a TS-doc  AND
                     it is corroborated (>=2 lanes) OR its RRF score clears a
                     margin over rank 2.
Heuristic, deterministic, defensible. An LLM critic is a documented stretch
goal (--llm-critic), not the default.

FINALIZE - apply the gated reranker (rung 7c''): rerank the fused pool, but if
the CE's best score < ce_floor the judge has collapsed, so keep the RRF order.
This is where the agent ROUTES AROUND the judge to keep a lane champion.

The agent reuses the exact tools already built and tested: dense via
store/embed, bm25_tool.ensure_index, rrf.rrf_fuse, rewrite.rewrite_query,
rerank.rerank. No retrieval logic is reinvented; the agent only orchestrates.

Standalone trace (mock tools, no services):  python eval\\agent.py --selftest
"""
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from rrf import rrf_fuse

TS_MARGIN = 1.3          # rank-1 RRF score >= TS_MARGIN * rank-2 => confident solo
CE_FLOOR = 0.5           # judge-collapse gate (same as run_eval --ce-floor)


def is_ts(label: str) -> bool:
    return label.startswith("TS-")


def _label(payload: dict) -> str:
    if payload.get("id"):
        return payload["id"]
    return f"spec:{payload.get('spec', '?')} {payload.get('clause', '?')}"


@dataclass
class Step:
    n: int
    action: str
    detail: str
    top_label: str
    sufficient: bool


@dataclass
class Trace:
    qid: str
    steps: list = field(default_factory=list)
    final_ranked: list = field(default_factory=list)   # list of labels (eval)
    final_hits: list = field(default_factory=list)      # list of hit objects (UI)
    tool_calls: int = 0
    llm_calls: int = 0
    stop_reason: str = ""
    rewrite_error: str = ""

    def log(self, **kw):
        self.steps.append(Step(**kw))


class Agent:
    """Orchestrates already-built tools. `tools` is a dict so tests inject mocks.
    Required keys: dense(query,k)->hits, bm25(query,k)->hits,
    rewrite(query)->str, rerank(query,hits)->[(hit,score)]. Each hit exposes
    .id and .payload (Qdrant ScoredPoint / BM25Hit both qualify)."""

    def __init__(self, tools, k=10, lane_k=10, budget=3,
                 ce_floor=CE_FLOOR, verbose=False):
        self.t = tools
        self.k = k
        self.lane_k = lane_k
        self.budget = budget
        self.ce_floor = ce_floor
        self.verbose = verbose

    # ---- reflection critic (NO cross-encoder score) -----------------------
    def _sufficient(self, fused, lane_ids):
        """fused: [(hit, rrf_score)] desc. lane_ids: {lane_name: set(ids)}.
        Sufficient iff the top hit is a TS-doc that is CORROBORATED - either by
        >=2 lanes agreeing (the autopsy's do+bo signal) or, when only one lane
        has run, by clearing a decisive RRF margin over rank 2.

        Design note (learned from the self-test's Q05 false positive): a lone
        TS-doc at rank 1 is NOT sufficient just because rank 2 is a spec. One
        lane's top pick is one lane's OPINION; a wrong TS-doc (TS-07 for Q05)
        looks structurally identical to a right one. Only cross-lane agreement
        distinguishes them, and the gold labels that would tell them apart are
        exactly what a real system lacks at query time. So single-lane results
        must seek a second lane (cheap, no LLM) rather than self-certify. Never
        reads the cross-encoder score (rungs 5/7c: it collapses on hard)."""
        if not fused:
            return False, "empty"
        top_hit, top_score = fused[0]
        top_lbl = _label(top_hit.payload or {})
        if not is_ts(top_lbl):
            return False, f"top is spec ({top_lbl})"
        lanes_hit = sum(1 for ids in lane_ids.values() if top_hit.id in ids)
        if lanes_hit >= 2:
            return True, f"{top_lbl} corroborated by {lanes_hit} lanes"
        if len(lane_ids) == 1:
            return False, f"{top_lbl} from 1 lane - seek corroboration"
        if len(fused) > 1:
            second_score = fused[1][1]
            if second_score > 0 and top_score >= TS_MARGIN * second_score:
                return True, f"{top_lbl} clears decisive margin over rank 2"
        return False, f"{top_lbl} uncorroborated across {len(lane_ids)} lanes"

    # ---- finalize: gated rerank (route around a collapsed judge) -----------
    def _finalize(self, query, fused, trace):
        pool = [h for h, _ in fused[:self.k]]
        judged = self.t["rerank"](query, pool)
        trace.tool_calls += 1
        ce_max = max((s for _, s in judged), default=0.0)
        if ce_max < self.ce_floor:
            ordered = [h for h, _ in fused[:self.k]]        # judge collapsed
            trace.stop_reason += f" | CE collapsed ({ce_max:.3f}) -> RRF order"
        else:
            ordered = [h for h, _ in judged[:self.k]]
            trace.stop_reason += f" | CE ok ({ce_max:.3f}) -> CE order"
        trace.final_hits = ordered
        return [_label(h.payload or {}) for h in ordered]

    # ---- the loop ----------------------------------------------------------
    def run(self, qid, query):
        tr = Trace(qid=qid)
        fused, lane_ids = [], {}

        # step 1 - dense only
        d = self.t["dense"](query, self.lane_k)
        tr.tool_calls += 1
        lane_ids = {"do": {h.id for h in d}}
        fused = rrf_fuse(d)
        ok, why = self._sufficient(fused, lane_ids)
        tr.log(n=1, action="dense_search", detail="original query",
               top_label=_label(fused[0][0].payload or {}) if fused else "-",
               sufficient=ok)
        if ok:
            tr.stop_reason = f"step1 dense sufficient: {why}"
            tr.final_ranked = self._finalize(query, fused, tr)
            return tr

        # step 2 - add BM25, fuse
        b = self.t["bm25"](query, self.lane_k)
        tr.tool_calls += 1
        lane_ids["bo"] = {h.id for h in b}
        fused = rrf_fuse(d, b)
        ok, why = self._sufficient(fused, lane_ids)
        tr.log(n=2, action="bm25_search", detail="fuse dense+bm25",
               top_label=_label(fused[0][0].payload or {}) if fused else "-",
               sufficient=ok)
        if ok:
            tr.stop_reason = f"step2 hybrid sufficient: {why}"
            tr.final_ranked = self._finalize(query, fused, tr)
            return tr

        # step 3 - rewrite, refetch both lanes, fuse everything (LLM: last).
        # A tool failing is a NORMAL operating condition for an agent; a timed-
        # out or empty rewrite must degrade to the step-2 hybrid result, never
        # crash the run. This is the graceful-degradation the loop requires.
        rw = None
        try:
            rw = self.t["rewrite"](query)
            if not rw or rw.strip() == query.strip():
                rw = None                       # no-op rewrite: nothing gained
        except Exception as e:                  # network/timeout/backend down
            tr.rewrite_error = repr(e)
            rw = None
        tr.tool_calls += 1
        if rw is None:
            tr.stop_reason = ("step3 rewrite unavailable "
                              f"({'no-op' if not getattr(tr,'rewrite_error',None) else 'tool error'})"
                              " -> kept step-2 hybrid result")
            tr.final_ranked = self._finalize(query, fused, tr)
            return tr

        tr.llm_calls += 1
        dr = self.t["dense"](rw, self.lane_k)
        br = self.t["bm25"](rw, self.lane_k)
        tr.tool_calls += 2
        lane_ids["dr"] = {h.id for h in dr}
        lane_ids["br"] = {h.id for h in br}
        fused = rrf_fuse(d, b, dr, br)
        ok, why = self._sufficient(fused, lane_ids)
        tr.log(n=3, action="rewrite+refetch",
               detail=f"rw='{rw[:40]}'", 
               top_label=_label(fused[0][0].payload or {}) if fused else "-",
               sufficient=ok)
        tr.stop_reason = (f"step3 budget spent ({'now sufficient: ' + why if ok else why})")
        tr.final_ranked = self._finalize(query, fused, tr)
        return tr


# --------------------------------------------------------------------------
# Self-test: mock tools, hand-traced escalation on three archetype queries.
# --------------------------------------------------------------------------
def _selftest():
    H = lambda i, lbl: type("H", (), {"id": i, "payload": ({"id": lbl} if is_ts(lbl)
                                      else {"spec": lbl.replace("spec:", ""),
                                            "clause": "x"})})()

    # Corpus of mock hits keyed by (which query text) -> ranked ids.
    # easy query "ping pong": dense nails TS-03 corroborated later.
    # Q14-like "carrier invisible": dense misses, bm25 catches TS-09.
    # Q05-like "expressway": only the rewrite lanes see TS-01.
    def make_tools(scenario):
        def dense(q, k):
            if scenario == "easy":
                return [H(1, "TS-03"), H(20, "spec:38.300")]
            if scenario == "bm25need":
                # dense sees TS-09 only weakly (rank 3), gold not at top
                return [H(99, "spec:38.331"), H(88, "TS-16"), H(9, "TS-09")]
            if scenario == "q05":
                if "handover" in q:  # the rewrite
                    return [H(1, "TS-01"), H(2, "TS-02")]
                return [H(7, "TS-07"), H(30, "spec:38.300")]  # original: wrong TS
            return []

        def bm25(q, k):
            if scenario == "easy":
                return [H(1, "TS-03"), H(50, "spec:38.413")]  # agrees -> 2 lanes
            if scenario == "bm25need":
                return [H(9, "TS-09"), H(40, "spec:38.300")]  # bm25 ranks it 1
            if scenario == "q05":
                if "handover" in q:
                    return [H(1, "TS-01"), H(60, "spec:38.300")]  # rewrite: gold
                return [H(20, "spec:38.133")]                     # original: none
            return []

        rewrite = lambda q: "too-late handover a3 offset radio link failure " + q[:10]
        rerank = lambda q, hits: [(h, 0.95) for h in hits]     # confident judge
        return {"dense": dense, "bm25": bm25, "rewrite": rewrite, "rerank": rerank}

    print("agent self-test (mock tools, hand-traced)\n" + "=" * 62)
    # easy: dense finds TS-03 but one lane can't self-certify -> one FREE step
    #       to bm25, which agrees -> stop at step 2, still 0 LLM calls.
    # bm25need: dense misses; bm25 supplies TS-09, and dense's spec-heavy list
    #       doesn't contest it, but corroboration needs the margin path here.
    # q05: no lane sees the gold until the rewrite -> full budget, 1 LLM call.
    expect = {
        "easy":     (2, 0, "TS-03"),   # cheap: 2 free steps, no LLM
        "bm25need": (2, 0, "TS-09"),   # cheap: bm25 rescues, no LLM
        "q05":      (3, 1, "TS-01"),   # expensive: full escalation, 1 LLM call
    }
    for scen, (exp_steps, exp_llm, exp_top) in expect.items():
        ag = Agent(make_tools(scen))
        tr = ag.run(scen.upper(), {"easy": "ping pong between cells",
                                   "bm25need": "strong carrier handsets ignore",
                                   "q05": "calls on the expressway die"}[scen])
        steps = len(tr.steps)
        top = tr.final_ranked[0] if tr.final_ranked else "-"
        tag = "OK" if (steps == exp_steps and tr.llm_calls == exp_llm
                       and top == exp_top) else "**FAIL**"
        print(f"\n[{scen}] {tag}  steps={steps} (exp {exp_steps})  "
              f"llm={tr.llm_calls} (exp {exp_llm})  top={top} (exp {exp_top})")
        for s in tr.steps:
            print(f"    step{s.n} {s.action:<16} top={s.top_label:<14} "
                  f"suff={s.sufficient}")
        print(f"    stop: {tr.stop_reason}")
        assert steps == exp_steps and tr.llm_calls == exp_llm and top == exp_top

    print("\nall archetypes escalated exactly as designed - "
          "easy stops cheap, Q05 spends the full budget once.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        ap.print_help()
