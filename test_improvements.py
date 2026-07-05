"""
test_improvements.py - end-to-end check of the 7 fixes, run against the LIVE engine.

It replays the exact conversation that failed in the console and asserts the
improvements. Run AFTER re-ingesting the 3 new scenarios and starting serve.py:

    python ingest_troubleshooting.py          # adds TS-17/18/19 (idempotent)
    curl -X POST http://localhost:8000/reload # rebuild BM25 (or restart serve.py)
    uvicorn serve:app --port 8000             # (if not already running)
    python test_improvements.py               # then this

For the frontier-routing checks to show 'openai:...', OPENAI_API_KEY must be set;
otherwise those turns fall back to local (the client Mermaid guard still applies).
"""
import json
import sys
import urllib.request

API = "http://localhost:8000"


def _post(ep, payload):
    req = urllib.request.Request(API + ep, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode())


def _get(ep):
    with urllib.request.urlopen(API + ep, timeout=30) as r:
        return json.loads(r.read().decode())


PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def hist(*pairs):
    """Build a history list from (user, assistant) pairs."""
    h = []
    for u, a in pairs:
        h.append({"role": "user", "content": u})
        h.append({"role": "assistant", "content": a})
    return h


def main():
    try:
        health = _get("/health")
    except Exception:
        sys.exit("Engine not reachable on :8000 — start it: uvicorn serve:app --port 8000")
    pts = health.get("points", health.get("count", "?"))
    print(f"Engine up. Corpus: {pts} points\n")

    # ---- #1 corpus gap closed: the 3 new scenarios are retrievable ----
    print("#1  Corpus gap — idle-reselection scenarios present & retrieved")
    d = _post("/ask", {"query": "A strong carrier on another NR frequency is visible but idle UEs never reselect to it",
                       "generate": False})
    labels = [c.get("label", "") for c in d.get("citations", [])]
    top10 = " ".join(labels[:10])
    hit_new = any(x in top10 for x in ("TS-17", "TS-18", "TS-19"))
    check("new idle-reselection scenario retrieved in top-10", hit_new,
          "top: " + ", ".join(labels[:6]))
    if not hit_new:
        print("      -> TS-17/18/19 not retrieved. Ingest them and rebuild the index:")
        print("           python ingest_troubleshooting.py")
        print("           curl -X POST http://localhost:8000/reload   (or restart serve.py)")
        print("         then re-run this test.")

    # ---- #4 no more refusal: partial grounded answer on the hard query ----
    print("\n#4  Partial-answer prompt — no outright refusal")
    d = _post("/ask", {"query": "A strong carrier is invisible to idle phones"})
    ans = (d.get("answer") or "").lower()
    refused = ("cannot formulate" in ans or "cannot answer" in ans
               or "does not directly address" in ans and "snonintra" not in ans)
    check("answers instead of refusing", not refused)
    check("names the real root cause (SnonIntraSearch / priority / Qrxlevmin)",
          any(k in ans for k in ("snonintrasearch", "reselectionpriority",
                                 "qrxlevmin", "priority")),
          "answer[:140]: " + (d.get("answer") or "")[:140].replace("\n", " "))

    # ---- #6 condenser strips 'Search for ...' wrappers on a follow-up ----
    print("\n#6  Condenser — clean standalone query on a follow-up")
    h = hist(("A strong carrier is invisible to idle phones",
              "Check SnonIntraSearchP/Q and cellReselectionPriority [1]."))
    d = _post("/ask", {"query": "can you construct command on this to troubleshoot",
                       "history": h})
    sq = d.get("search_query", "")
    check("condensed follow-up is set", bool(d.get("condensed")), f"search_query: {sq[:80]}")
    check("no 'search for' / stray-quote artifact in the query",
          not sq.lower().startswith(("search", "find", "look")) and '"' not in sq[:2])

    # ---- #2 intent routing: synthesis/command turns go frontier ----
    print("\n#2  Frontier routing — hard synthesis auto-routes to the frontier model")
    for q in ["give the call flow to identify the issue",
              "can you construct command on this to troubleshoot",
              "possible gnodeb is ericsson give troubleshooting commands"]:
        d = _post("/ask", {"query": q, "history": h})
        routed = d.get("routed_frontier")
        mode = d.get("generation_mode", "")
        check(f"routed_frontier flag set :: {q[:44]}", bool(routed), f"mode={mode}")

    # ---- #3 no-command hallucination: command request stays honest ----
    print("\n#3  No-command guard — commands not fabricated from specs")
    d = _post("/ask", {"query": "can you construct command on this to troubleshoot", "history": h})
    ans = (d.get("answer") or "")
    low = ans.lower()
    fabricated = "gnb#" in low or "gnb #" in low or "show cell" in low
    honest = any(k in low for k in ("vendor", "not contain", "out of scope",
                                    "specification", "parameter"))
    check("does NOT fabricate gNB CLI from specs", not fabricated)
    check("honestly points to parameters / vendor scope", honest,
          "answer[:150]: " + ans[:150].replace("\n", " "))

    # ---- #5 web-source flagging (only checkable if a web result is present) ----
    print("\n#5  Web-source flag — informational (needs a /research turn to trigger)")
    print("      run a Deep Research query, then confirm the answer flags 'unverified web source'.")

    # ---- summary ----
    print("\n" + "=" * 60)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("Failed:", ", ".join(FAIL))
    print("=" * 60)
    print("\nManual review (quality, not pass/fail):")
    print(" - Re-run the full 5-turn conversation in the console and confirm the")
    print("   call-flow/command turns now carry a ⚡ frontier badge and read better.")
    print(" - Generate a 'Flow' artifact for the fault and confirm the Mermaid renders.")


if __name__ == "__main__":
    main()
