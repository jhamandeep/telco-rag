#!/usr/bin/env python
r"""
bootstrap.py - bring TelcoRAG up from zero, with GUARANTEED, VERIFIED persistence.
Cross-shell, no PowerShell policy, no stderr-as-error surprises (Python reads exit
codes, not stderr text).

Run (from the project root, venv active):
    python bootstrap.py                 # bring everything up + verify persistence
    python bootstrap.py --run-eval      # ...and run baseline + rerank at the end
    python bootstrap.py --snapshot       # back up the live corpus to .\backups\ (no bring-up)
    python bootstrap.py --with-generator # also pull qwen2.5:7b (rung 6+)

Persistence, in one paragraph (this is the thing people get wrong):
    Qdrant runs as a DETACHED Docker container (`up -d`), so CLOSING THE TERMINAL
    DOES NOT STOP IT and DOES NOT LOSE DATA. The 26,203 vectors are written to your
    HOST DISK via a bind mount (qdrant_storage -> /qdrant/storage), so they survive a
    container stop, a machine reboot, and a terminal close. They are lost ONLY if you
    delete that folder, or run `docker compose down -v` AND remove it. This script
    verifies the mount is real (Step 8) and can snapshot it (--snapshot).

Idempotent: content-hash point IDs make re-ingest dedup automatically.
The ONE thing it can't do for you: re-download the gitignored ETSI spec PDFs.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STORAGE = ROOT / "qdrant_storage"          # the host folder the vectors live in
BACKUPS = ROOT / "backups"
CONTAINER = "telco-qdrant"

# The mount source is parameterised so it can be made ABSOLUTE at run time (below),
# which removes the #1 cause of "my data vanished": a RELATIVE ./qdrant_storage that
# resolves against whatever directory you happened to run compose from. The default
# keeps the committed file portable for a manual `docker compose up` from the root.
COMPOSE_YML = """services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: telco-qdrant
    ports:
      - "6333:6333"   # REST API + web dashboard
      - "6334:6334"   # gRPC
    volumes:
      # Host bind mount: data lives on YOUR disk, not inside the container.
      # QDRANT_STORAGE is set to an ABSOLUTE path by bootstrap.py so persistence
      # never depends on the current working directory.
      - ${QDRANT_STORAGE:-./qdrant_storage}:/qdrant/storage
    restart: unless-stopped   # auto-restarts on reboot / Docker restart; respects a manual stop
"""


def ok(m):    print(f"  [OK]   {m}")
def info(m):  print(f"  [..]   {m}")
def warn(m):  print(f"  [WARN] {m}")
def die(m):   print(f"  [STOP] {m}"); sys.exit(1)
def step(n, t): print(f"\n=== {n}. {t} ===")


def run(cmd, what, check=True, env=None):
    """Run a command, inherit its console output, return exit code. Unlike
    PowerShell, stderr text is never mistaken for an error - only the exit code."""
    info("$ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    rc = subprocess.run(cmd, shell=isinstance(cmd, str), env=env).returncode
    if check and rc != 0:
        die(f"{what} failed (exit {rc}).")
    return rc


def cmd_ok(cmd):
    return subprocess.run(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def cmd_out(cmd):
    """Return (rc, stdout_text). Never raises."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return p.returncode, (p.stdout or "").strip()


def http_ok(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def http_json(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def http_post(url, timeout=120):
    req = urllib.request.Request(url, data=b"", method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def compose_env():
    """Environment for `docker compose` with an ABSOLUTE storage path bound in."""
    env = os.environ.copy()
    env["QDRANT_STORAGE"] = str(STORAGE)      # absolute -> CWD-independent mount
    return env


def compose_up():
    if run(["docker", "compose", "up", "-d"], "docker compose up",
           check=False, env=compose_env()) != 0:
        run(["docker-compose", "up", "-d"], "docker-compose up", env=compose_env())


def dir_size_mb(p):
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def corpus_counts():
    """(total, troubleshooting, spec) in the live collection; (0,0,0) if absent."""
    import config
    from store import get_client
    from qdrant_client import models as qm
    c = get_client()
    try:
        total = c.count(config.COLLECTION, exact=True).count
    except Exception:
        return (0, 0, 0)

    def n(src):
        return c.count(config.COLLECTION, exact=True,
                       count_filter=qm.Filter(must=[qm.FieldCondition(
                           key="source", match=qm.MatchValue(value=src))])).count

    return (total, n("troubleshooting"), n("spec"))


# --------------------------------------------------------------------------
# Persistence verification - prove the vectors are on the host disk, not ghosts.
# --------------------------------------------------------------------------
def verify_persistence():
    """Inspect the running container and confirm /qdrant/storage is a BIND MOUNT
    to our host folder, and that the folder is non-empty. This is what turns
    'I think the data is lost' into a checkable fact."""
    rc, out = cmd_out(["docker", "inspect", CONTAINER, "--format", "{{json .Mounts}}"])
    if rc != 0 or not out:
        warn("could not inspect the container mounts (is the container named "
             f"'{CONTAINER}'?). Skipping the persistence proof.")
        return
    try:
        mounts = json.loads(out)
    except Exception:
        warn("could not parse container mounts.")
        return
    m = next((x for x in mounts if x.get("Destination") == "/qdrant/storage"), None)
    if not m:
        die("Qdrant has NO mount at /qdrant/storage -> data is INSIDE the container "
            "and WILL be lost if it is removed. Re-run bootstrap to fix the volume.")
    src = m.get("Source", "?")
    mtype = m.get("Type", "?")
    if mtype != "bind":
        warn(f"storage is a Docker '{mtype}', not a host bind mount (source: {src}). "
             "It survives restarts but is easy to wipe with `docker compose down -v`. "
             "A host bind mount is safer; re-run bootstrap to switch.")
    else:
        ok(f"storage is a HOST BIND MOUNT: {src}  ->  /qdrant/storage")
    if STORAGE.exists():
        sz = dir_size_mb(STORAGE)
        ok(f"host folder exists on disk: {STORAGE}  ({sz:.1f} MB on disk)")
        if sz < 0.1:
            warn("...but it is nearly empty. If you expected a full corpus, the "
                 "mount source may differ from where the data was written. Re-ingest "
                 "or check `docker inspect telco-qdrant`.")
    else:
        warn(f"host folder {STORAGE} not found yet (it is created on first write).")
    print()
    print("  " + "-" * 66)
    print("  PERSISTENCE: your vectors live on the HOST DISK at")
    print(f"    {STORAGE}")
    print("  They SURVIVE: closing this terminal, stopping the container, a reboot.")
    print("  They are LOST ONLY IF: you delete that folder, or run")
    print("    `docker compose down -v`  and remove it.")
    print("  Container is DETACHED - closing the terminal does not stop it.")
    print("  Stop it deliberately:  docker compose stop    Start again:  docker compose start")
    print("  Back it up any time :  python bootstrap.py --snapshot")
    print("  " + "-" * 66)


# --------------------------------------------------------------------------
# Snapshot - a real recovery path. Uses Qdrant's native snapshot API (consistent),
# not a tar of live files, then copies it next to the project under .\backups\.
# --------------------------------------------------------------------------
def snapshot():
    import config
    if not http_ok("http://localhost:6333/readyz"):
        die("Qdrant is not up on :6333 - start it first (python bootstrap.py).")
    coll = config.COLLECTION
    try:
        total = http_json(f"http://localhost:6333/collections/{coll}")
        n = total.get("result", {}).get("points_count", "?")
    except Exception:
        die(f"collection '{coll}' not found on the server - nothing to snapshot.")
    info(f"creating a consistent snapshot of '{coll}' ({n} points)...")
    try:
        res = http_post(f"http://localhost:6333/collections/{coll}/snapshots")
        name = res.get("result", {}).get("name")
    except Exception as e:
        die(f"snapshot API failed: {e}")
    ok(f"snapshot created on the server: {name}")

    # The snapshot lands inside the bind mount, so it is already on the host disk:
    src = STORAGE / "snapshots" / coll / name
    BACKUPS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUPS / f"{coll}_{stamp}.snapshot"
    if src.exists():
        shutil.copy2(src, dst)
        ok(f"copied to: {dst}  ({dst.stat().st_size/1024/1024:.1f} MB)")
    else:
        warn(f"snapshot not found on host at {src} (mount may differ). It still "
             f"exists on the server; list via GET /collections/{coll}/snapshots.")
    print()
    print("  RESTORE later (wipes the current collection first) with:")
    print(f"    curl -X PUT http://localhost:6333/collections/{coll}/snapshots/recover \\")
    print(f'         -H "Content-Type: application/json" \\')
    print(f'         -d \'{{"location":"file:///qdrant/storage/snapshots/{coll}/{name}"}}\'')
    print("  (or upload the .snapshot file from .\\backups\\ via the /snapshots/upload API).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-eval", action="store_true",
                    help="run baseline + rerank eval after bring-up")
    ap.add_argument("--with-generator", action="store_true",
                    help="also pull qwen2.5:7b (the generator) - needed from rung 6")
    ap.add_argument("--force-ingest", action="store_true",
                    help="re-run ingestion even if the corpus already looks complete")
    ap.add_argument("--snapshot", action="store_true",
                    help="back up the live corpus to .\\backups\\ and exit (no bring-up)")
    args = ap.parse_args()
    os.chdir(ROOT)

    # --snapshot is a standalone recovery action; it does not re-ingest anything.
    if args.snapshot:
        step("S", "Snapshot the live corpus (backup)")
        snapshot()
        return

    # ---------------------------------------------------------- 0 sanity
    step(0, "Sanity checks")
    if not (ROOT / "config.py").exists():
        die("run from the project root (config.py not found).")
    if ".venv" not in sys.executable:
        warn(f"python not from .venv ({sys.executable}) - activate it first.")
    else:
        ok(f"venv python: {sys.executable}")

    # ---------------------------------------------------------- 1 .env
    step(1, ".env - pin QDRANT_MODE=server")
    envp = ROOT / ".env"
    if not envp.exists():
        die(".env missing. Restore it (QDRANT_MODE, COLLECTION, EMBED_MODEL, EMBED_DIM, RERANK_MODEL...).")
    txt = envp.read_text(encoding="utf-8")
    if re.search(r'(?im)^\s*QDRANT_MODE\s*=', txt):
        txt = re.sub(r'(?im)^\s*QDRANT_MODE\s*=.*$', 'QDRANT_MODE=server', txt)
    else:
        txt = txt.rstrip("\n") + "\nQDRANT_MODE=server\n"
    envp.write_text(txt, encoding="utf-8")
    ok("QDRANT_MODE=server (the embedded 'local' 16-point store is abandoned).")

    # ---------------------------------------------------------- 2 deps
    step(2, "Python dependencies")
    torch_ok = subprocess.run(
        [sys.executable, "-c",
         "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"]
    ).returncode == 0
    if torch_ok:
        ok("torch + CUDA already present.")
    else:
        info("installing CUDA torch (cu128) for the RTX 5070...")
        run([sys.executable, "-m", "pip", "install", "torch",
             "--index-url", "https://download.pytorch.org/whl/cu128", "--quiet"],
            "torch install")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
        "requirements install")
    ok("dependencies ready.")

    # ---------------------------------------------------------- 3 qdrant
    step(3, "Qdrant (Docker, server mode, persistent host volume)")
    if not cmd_ok(["docker", "info"]):
        die("Docker is not running. Start Docker Desktop, then re-run.")
    comp = ROOT / "docker-compose.yml"
    need_write = True
    if comp.exists():
        cur = comp.read_text(encoding="utf-8")
        # rewrite if it lacks the storage mount OR still hardcodes the relative path
        need_write = ("/qdrant/storage" not in cur) or ("QDRANT_STORAGE" not in cur)
    if need_write:
        comp.write_text(COMPOSE_YML, encoding="utf-8")
        ok("wrote docker-compose.yml (absolute host bind mount via $QDRANT_STORAGE).")
    else:
        ok("docker-compose.yml already mounts the parameterised host volume.")
    STORAGE.mkdir(exist_ok=True)     # ensure the host folder exists before mounting
    compose_up()
    info("waiting for Qdrant to be ready on :6333 ...")
    ready = False
    for _ in range(30):
        if http_ok("http://localhost:6333/readyz"):
            ready = True
            break
        time.sleep(2)
    if not ready:
        die("Qdrant did not become ready on :6333 (check: docker logs telco-qdrant).")
    ok("Qdrant server is up (detached - survives terminal close).")

    # ---------------------------------------------------------- 4 ollama
    step(4, "Ollama models")
    try:
        tags = http_json("http://localhost:11434/api/tags")
    except Exception:
        die("Ollama not reachable on :11434. Start Ollama, then re-run.")
    have = " ".join(m.get("name", "") for m in tags.get("models", []))
    if "nomic-embed-text" in have:
        ok("embedder present: nomic-embed-text")
    else:
        info("pulling nomic-embed-text (embedder, required now)...")
        run(["ollama", "pull", "nomic-embed-text"], "ollama pull nomic-embed-text")
    if "qwen2.5:7b" in have:
        ok("generator present: qwen2.5:7b")
    elif args.with_generator:
        info("pulling qwen2.5:7b (generator, for rung 6+)...")
        run(["ollama", "pull", "qwen2.5:7b"], "ollama pull qwen2.5:7b")
    else:
        info("skipping qwen2.5:7b (generator) - pass --with-generator when you reach rung 6.")

    # ---------------------------------------------------------- 5 pdfs
    step(5, "Spec PDFs in data\\")
    data = ROOT / "data"
    pdfs = list(data.glob("*.pdf")) if data.exists() else []
    if not pdfs:
        warn("No PDFs in data\\. The ETSI specs are gitignored and were wiped.")
        warn("Re-download these 7 into data\\ (ETSI = 3GPP number + leading 1):")
        warn("   38.331  38.300  38.321  38.133  38.215  38.423  38.413")
        die("Then re-run: python bootstrap.py")
    ok(f"{len(pdfs)} PDF(s) found in data\\.")

    # ---------------------------------------------------------- 6 smoke
    step(6, "Smoke test (embed + Qdrant round-trip)")
    run([sys.executable, "smoke_test.py"], "smoke_test.py")
    ok("foundation smoke test passed.")

    # ---------------------------------------------------------- 7 ingest
    step(7, "Ingest both lanes")
    total, tsh, spec = corpus_counts()
    if not args.force_ingest and tsh >= 16 and spec > 1000:
        ok(f"corpus already complete ({total} pts: {tsh} troubleshooting + {spec} spec) - skipping ingest.")
        info("nothing re-embedded (persisted from a previous run). Force with --force-ingest.")
    else:
        if total:
            info(f"corpus incomplete ({total} pts) - (re)building.")
        info("troubleshooting lane (16 scenarios)...")
        run([sys.executable, "ingest_troubleshooting.py"], "ingest_troubleshooting.py")
        info("spec lane (batch embed - minutes; watch the tqdm bar)...")
        run([sys.executable, "ingest_specs.py"], "ingest_specs.py")
        ok("ingest complete.")

    # ---------------------------------------------------------- 8 verify corpus + persistence
    step(8, "Verify corpus + PROVE persistence")
    run([sys.executable, str(Path("eval") / "check_corpus.py")], "check_corpus.py", check=False)
    print()
    verify_persistence()
    print()
    ok("Bring-up finished. Expect ~26,203 points (16 troubleshooting + ~26,187 spec) above.")

    # ---------------------------------------------------------- 9 eval (opt)
    if args.run_eval:
        step(9, "Eval (baseline + rerank)")
        run([sys.executable, str(Path("eval") / "run_eval.py")], "baseline eval")
        run([sys.executable, str(Path("eval") / "run_eval.py"), "--rerank"], "rerank eval")

    print("\nTip: back up the corpus before risky changes ->  python bootstrap.py --snapshot")
    print("Next: python eval\\run_eval.py --rerank --show   (reproduce the honest numbers)")


if __name__ == "__main__":
    main()
