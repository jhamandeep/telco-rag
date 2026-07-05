# TelcoRAG - single-workstation build

Agentic RAG over 3GPP RAN / mobility specs (RRC + handover), running fully local
on one box: Windows 11 + RTX 5070 (12 GB). Ollama (GPU) for embed + generate,
Qdrant embedded for vectors, a CUDA cross-encoder reranker (added later).

## Prerequisites
- Python 3.10+, current NVIDIA driver for the RTX 5070.
- Ollama for Windows: https://ollama.com/download (auto-detects the 5070).
- Docker is OPTIONAL (only if you set QDRANT_MODE=server).

## Setup (PowerShell, inside this folder)
```powershell
# 1. Models (Ollama, native - talks to the GPU directly)
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

# 2. Python project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env

# 3. Verify the foundation (Qdrant runs EMBEDDED - no Docker)
python smoke_test.py
```
Expected tail: `Foundation OK - embedder + vector store verified on this box.`

## Vector store modes
- `QDRANT_MODE=local` (default): embedded in the Python process, persisted to
  `./qdrant_storage`. No Docker.
- `QDRANT_MODE=server`: set it in `.env`, then `docker compose up -d`, and Qdrant
  runs as a container at http://localhost:6333 (with a web dashboard).

## Files
| File | Role |
|---|---|
| `config.py` | Single source of truth - endpoints, models, the switches. Reads `.env`. |
| `store.py` | Returns the Qdrant client (embedded or server) per `QDRANT_MODE`. |
| `.env` / `.env.example` | Settings + secrets. `.env` is gitignored - never commit it. |
| `docker-compose.yml` | Qdrant server (only used when QDRANT_MODE=server). |
| `smoke_test.py` | Proves Ollama + Qdrant talk on this box. |
| `chunker.py` | Spec-structure-aware chunker (fixed baseline + structure mode). |
| `requirements.txt` | Foundation deps. torch (CUDA) + reranker + RAGAS added per phase. |

## Next milestones
1. Now: foundation up + smoke_test.py passes.
2. Ingest: PDF -> chunker.py -> embed -> upsert into telco_ran.
3. Baseline retrieval + gold eval set (P@K, R@K, MRR).
4. Structure-aware re-ingest -> the chunking ablation.
5. Reranker + agentic loop + faithfulness critic.
6. RAGAS + report + video.
