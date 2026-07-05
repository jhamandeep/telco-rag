"""
config.py - single source of truth for the TelcoRAG stack (single box).

Every value comes from the environment / .env (which is gitignored).
Never hardcode secrets here.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- endpoints ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
COLLECTION = os.getenv("COLLECTION", "telco_ran")

# --- vector store: embedded (no Docker) or server ---
QDRANT_MODE = os.getenv("QDRANT_MODE", "local")               # "local" | "server"
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")    # used when QDRANT_MODE=local
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333") # used when QDRANT_MODE=server

# --- embedder (Ollama, GPU) ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768  # nomic-embed-text output dimension

# --- generator: config-switchable (local 5070 <-> OpenAI) ---
GENERATOR = os.getenv("GENERATOR", "local")          # "local" | "openai"
LOCAL_GEN_MODEL = os.getenv("LOCAL_GEN_MODEL", "qwen2.5:7b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")         # only needed if GENERATOR=openai

# --- reranker (CUDA PyTorch, native; wired in the rerank phase) ---
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# --- retrieval / chunking params ---
TOP_K = int(os.getenv("TOP_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
CHUNK_MODE = os.getenv("CHUNK_MODE", "structure")    # "fixed" | "structure"
