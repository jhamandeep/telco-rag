"""
rewrite.py - LLM query rewrite (rung 6), switchable between a local and an
OpenAI generator so we can compare them in the report.

Why rewrite: the red-zone misses (e.g. Q05) fail because the symptom-language
query and its RAN-engineer answer doc share almost no vocabulary, so the embedder
places them far apart and the gold never enters the fetched top-K. An LLM rewrite
into technical vocabulary moves the query vector next to the answer doc; a fresh
search then returns a different top-K that can contain it. Reranking can't do this.

Backend & models come from config.py (single source of truth), which reads .env:
  GENERATOR        'local' (Ollama) or 'openai'      <- the switch
  LOCAL_GEN_MODEL  e.g. qwen2.5:7b                    (temp 0 -> deterministic)
  OPENAI_MODEL     e.g. gpt-5.5                        (reasoning model: no temperature)
  OPENAI_API_KEY   your key                            (only if GENERATOR=openai)
Set REWRITE_BACKEND=local|openai to override GENERATOR for the rewrite step only.

CLI - see one rewrite on whichever backend is configured:
    python eval\\rewrite.py "Calls from cars die where one sector ends"
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

OLLAMA_GEN = config.OLLAMA_URL + "/api/generate"
OLLAMA_TAGS = config.OLLAMA_URL + "/api/tags"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# PROMPT_TAG is recorded in every results file so runs made with different
# prompts are never silently compared. v2: forbids CamelCase/comma formats -
# 'SCGFailure' tokenizes to one dead token and silently breaks the BM25 lane.
PROMPT_TAG = "v2"

INSTRUCTION = (
    "You are a 5G RAN troubleshooting expert. Rewrite the user's plain-language "
    "symptom report as a concise technical search query using standard 3GPP terms "
    "(e.g. handover, too-late or too-early handover, radio link failure, measurement "
    "report, A3 offset, time-to-trigger, RACH, beam failure FR2, measurement gaps, "
    "inter-frequency neighbour, Xn, path switch, SCG failure, uplink coverage). "
    "Pick only the terms that fit THIS symptom - do not list unrelated ones. "
    "Format: plain lowercase words separated by single spaces. Never CamelCase, "
    "never concatenated words, never commas (write 'scg failure', not "
    "'SCGFailure'). "
    "Output ONLY the rewritten query on a single line - no preamble, no quotes."
)


def _backend():
    # REWRITE_BACKEND env overrides config.GENERATOR for the rewrite step only.
    return (os.getenv("REWRITE_BACKEND") or config.GENERATOR or "local").lower()


def active_generator():
    """Label recorded in the results file so local vs OpenAI runs are distinct."""
    if _backend() == "openai":
        return "openai:" + config.OPENAI_MODEL
    return "ollama:" + config.LOCAL_GEN_MODEL


# ---- local (Ollama) --------------------------------------------------------
def _available_models():
    with urllib.request.urlopen(OLLAMA_TAGS, timeout=5) as r:
        return [m.get("name", "") for m in json.loads(r.read().decode()).get("models", [])]


def _resolve_model(model):
    avail = _available_models()
    if model in avail:
        return model
    match = next((m for m in avail if m.startswith(model) or model in m), None)
    if match:
        return match
    raise SystemExit(
        f"Generator '{model}' is not in Ollama.\n  Available: {', '.join(avail) or '(none)'}\n"
        f"  Fix: ollama pull {model}  (or set LOCAL_GEN_MODEL in .env)")


def _rewrite_ollama(query, timeout):
    model = _resolve_model(config.LOCAL_GEN_MODEL)
    body = json.dumps({
        "model": model,
        "prompt": f"{INSTRUCTION}\n\nSymptom: {query}\nTechnical query:",
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_GEN, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()).get("response", "").strip()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Ollama HTTP {e.code} for '{model}': {e.read().decode(errors='replace')}")


# ---- OpenAI ----------------------------------------------------------------
def _rewrite_openai(query, timeout):
    key = config.OPENAI_API_KEY or ""
    if not key or "REPLACE" in key:
        raise SystemExit("OPENAI_API_KEY not set. Add it to .env: OPENAI_API_KEY=sk-...")
    model = config.OPENAI_MODEL
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": f"Symptom: {query}\nTechnical query:"},
        ],
        # no temperature: gpt-5.x reasoning models reject non-default values.
    }).encode("utf-8")
    req = urllib.request.Request(OPENAI_URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"OpenAI HTTP {e.code} for model '{model}': "
                         f"{e.read().decode(errors='replace')}")
    return data["choices"][0]["message"]["content"].strip()


# ---- public ----------------------------------------------------------------
def rewrite_query(query, timeout=120):
    """Return a technical-vocabulary rewrite via the configured backend.
    Falls back to the original query if the model returns nothing."""
    resp = _rewrite_openai(query, timeout) if _backend() == "openai" \
        else _rewrite_ollama(query, timeout)
    first = next((ln.strip().strip('"') for ln in resp.splitlines() if ln.strip()), "")
    return first or query


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: python eval\\rewrite.py "your symptom query"')
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    print("backend  :", active_generator())
    print("original :", q)
    print("rewritten:", rewrite_query(q))
