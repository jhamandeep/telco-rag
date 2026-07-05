#Requires -Version 5
<#
  bootstrap.ps1 - bring TelcoRAG up from zero on Windows.

  Fixes the exact failure that just bit you: after a system/Docker reset the
  app was reading the embedded 'local' Qdrant (16 points, no specs). This
  script pins QDRANT_MODE=server, starts Qdrant with a PERSISTENT volume so a
  restart can't wipe it again, verifies Ollama, re-ingests both lanes, and
  confirms the corpus is ~26,203 before you trust any number.

  Run from the project root, in the activated venv:
      .\.venv\Scripts\Activate.ps1
      .\bootstrap.ps1               # bring everything up
      .\bootstrap.ps1 -RunEval      # ...and run the baseline + rerank eval at the end

  Idempotent: content-hash point IDs make re-ingest dedup automatically, so
  running this twice is harmless.

  The ONE thing it cannot do for you: re-download the ETSI spec PDFs (they are
  gitignored and were wiped). If data\ has no PDFs it will stop and tell you.
#>
param([switch]$RunEval)

$ErrorActionPreference = "Stop"

function Ok  ($m){ Write-Host "  [OK]   $m" -ForegroundColor Green }
function Info($m){ Write-Host "  [..]   $m" -ForegroundColor Cyan }
function Warn($m){ Write-Host "  [WARN] $m" -ForegroundColor Yellow }
function Die ($m){ Write-Host "  [STOP] $m" -ForegroundColor Red; exit 1 }
function Step($n,$t){ Write-Host "`n=== $n. $t ===" -ForegroundColor White }
# native commands don't throw in PowerShell - check exit code explicitly
function CheckExit($what){ if ($LASTEXITCODE -ne 0){ Die "$what failed (exit $LASTEXITCODE)." } }

# ---------------------------------------------------------------- 0. sanity
Step 0 "Sanity checks"
if (-not (Test-Path ".\config.py")) { Die "run from the project root (config.py not found)." }
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py)            { Die "python not on PATH. Activate the venv first." }
elseif ($py -notmatch "\.venv") { Warn "python is not from .venv ($py). Activate: .\.venv\Scripts\Activate.ps1" }
else                    { Ok "venv python: $py" }

# ---------------------------------------------------------------- 1. .env
Step 1 ".env - pin QDRANT_MODE=server"
if (-not (Test-Path ".\.env")) {
    Die ".env missing. Restore it (needs QDRANT_MODE, COLLECTION, EMBED_MODEL, EMBED_DIM, RERANK_MODEL...)."
}
$envLines = Get-Content ".\.env"
if ($envLines -match '^\s*QDRANT_MODE\s*=') {
    $envLines = $envLines -replace '^\s*QDRANT_MODE\s*=.*', 'QDRANT_MODE=server'
} else {
    $envLines += 'QDRANT_MODE=server'
}
Set-Content ".\.env" $envLines
Ok "QDRANT_MODE=server (the embedded 'local' 16-point store is abandoned)."

# ---------------------------------------------------------------- 2. deps
Step 2 "Python dependencies"
# torch first, from the cu128 index (RTX 5070 / sm_120), only if missing
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
if ($LASTEXITCODE -eq 0) {
    Ok "torch + CUDA already present."
} else {
    Info "installing CUDA torch (cu128) for the RTX 5070..."
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --quiet
    CheckExit "torch install"
}
Info "pip install -r requirements.txt (qdrant-client, pymupdf, sentence-transformers, ...)"
python -m pip install -r requirements.txt --quiet
CheckExit "requirements install"
Ok "dependencies ready."

# ---------------------------------------------------------------- 3. Qdrant
Step 3 "Qdrant (Docker, server mode, persistent)"
docker info *> $null
if ($LASTEXITCODE -ne 0) { Die "Docker is not running. Start Docker Desktop, then re-run." }

if (-not (Test-Path ".\docker-compose.yml")) {
    @'
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: telco_qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    restart: unless-stopped
'@ | Set-Content ".\docker-compose.yml"
    Ok "wrote docker-compose.yml with a persistent host volume (./qdrant_storage)."
} else {
    $compose = Get-Content ".\docker-compose.yml" -Raw
    if ($compose -notmatch "qdrant/storage") {
        Warn "docker-compose.yml has no persistent volume - a Docker reset can wipe data again."
        Warn "Add under the qdrant service:  volumes:`n           - ./qdrant_storage:/qdrant/storage"
    } else {
        Ok "docker-compose.yml already mounts a persistent volume."
    }
}

Info "starting Qdrant..."
docker compose up -d 2>$null
if ($LASTEXITCODE -ne 0) { docker-compose up -d; CheckExit "docker compose up" }

Info "waiting for Qdrant to be ready on :6333 ..."
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:6333/readyz" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { Start-Sleep -Seconds 2 }
}
if (-not $ready) { Die "Qdrant did not become ready on :6333 (check 'docker logs telco_qdrant')." }
Ok "Qdrant server is up."

# ---------------------------------------------------------------- 4. Ollama
Step 4 "Ollama models"
try { $tags = Invoke-RestMethod "http://localhost:11434/api/tags" -TimeoutSec 3 }
catch { Die "Ollama not reachable on :11434. Start Ollama, then re-run." }
$have = @($tags.models.name)
foreach ($mdl in @("nomic-embed-text", "qwen2.5:7b")) {
    if ($have -match [regex]::Escape($mdl)) {
        Ok "model present: $mdl"
    } else {
        Info "pulling $mdl ..."
        ollama pull $mdl
        CheckExit "ollama pull $mdl"
    }
}

# ---------------------------------------------------------------- 5. PDFs
Step 5 "Spec PDFs in data\"
$pdfs = @(Get-ChildItem ".\data\*.pdf" -ErrorAction SilentlyContinue)
if ($pdfs.Count -eq 0) {
    Write-Host ""
    Warn "No PDFs in data\. The ETSI specs are gitignored and were wiped."
    Warn "Re-download these 7 (ETSI number = 3GPP number + leading 1) into data\ :"
    Warn "   38.331  38.300  38.321  38.133  38.215  38.423  38.413"
    Die  "Then re-run .\bootstrap.ps1"
}
Ok "$($pdfs.Count) PDF(s) found in data\."

# ---------------------------------------------------------------- 6. smoke
Step 6 "Smoke test (embed + Qdrant round-trip)"
python smoke_test.py
CheckExit "smoke_test.py"
Ok "foundation smoke test passed."

# ---------------------------------------------------------------- 7. ingest
Step 7 "Ingest both lanes"
Info "troubleshooting lane (16 scenarios)..."
python ingest_troubleshooting.py
CheckExit "ingest_troubleshooting.py"
Info "spec lane (batch embed - minutes, watch the tqdm bar)..."
python ingest_specs.py
CheckExit "ingest_specs.py"
Ok "ingest complete."

# ---------------------------------------------------------------- 8. verify
Step 8 "Verify corpus"
python eval\check_corpus.py
Write-Host ""
Ok "Bring-up finished. Expect ~26,203 points (16 troubleshooting + ~26,187 spec) above."

# ---------------------------------------------------------------- 9. eval (opt)
if ($RunEval) {
    Step 9 "Eval (baseline + rerank)"
    python eval\run_eval.py
    CheckExit "baseline eval"
    python eval\run_eval.py --rerank
    CheckExit "rerank eval"
}

Write-Host "`nNext:  python eval\run_eval.py --rerank --show   (should reproduce yesterday's honest numbers)" -ForegroundColor White
