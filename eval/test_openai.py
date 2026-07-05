"""
test_openai.py - verify your OpenAI key + model before running the full eval on
the paid API. Forces the OpenAI backend and does one rewrite.

Setup (.env at project root):
    OPENAI_API_KEY=sk-...your key...
    OPENAI_MODEL=gpt-5.5

Run (works whether this file sits in eval\\ or the project root):
    python eval\\test_openai.py
"""
import os
import sys
from pathlib import Path

# rewrite.py lives in eval\. Find it whether this file is in eval\ or root.
_here = Path(__file__).resolve().parent
_evaldir = next((c for c in (_here, _here / "eval", _here.parent)
                 if (c / "rewrite.py").exists()), _here)
sys.path.insert(0, str(_evaldir))                                  # so 'rewrite' resolves
sys.path.append(str(_evaldir.parent if _evaldir.name == "eval" else _evaldir))  # so 'config' resolves

import config  # loads .env
os.environ["REWRITE_BACKEND"] = "openai"   # force OpenAI just for this test
from rewrite import rewrite_query, active_generator

Q = ("Calls from cars on the expressway die right where one sector ends and "
     "the next begins; coverage plots look clean.")

print("backend  :", active_generator())
print("original :", Q)
print("rewritten:", rewrite_query(Q))
print("\nOK - key and model work. To run the full eval on OpenAI:")
print("  set REWRITE_BACKEND=openai in .env  (or  $env:REWRITE_BACKEND=\"openai\")")
print("  then: python eval\\run_eval.py --rewrite")
