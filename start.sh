#!/bin/bash
set -e

cd "$(dirname "$0")"

# Download prebuilt artifacts from Hugging Face Hub if they aren't already present.
# HF_REPO_ID must be set (e.g. your-username/astrophysics-rag-artifacts).
if [ -n "$HF_REPO_ID" ]; then
    python3 - <<'PYEOF'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = os.environ["HF_REPO_ID"]
base = Path(__file__).parent.resolve() if "__file__" in dir() else Path(".")

files = [
    ("chunked_papers.json",          Path("chunked_papers.json")),
    ("faiss_index/astrophysics.index", Path("faiss_index/astrophysics.index")),
    ("faiss_index/metadata.pkl",     Path("faiss_index/metadata.pkl")),
    ("faiss_index/config.json",      Path("faiss_index/config.json")),
]

for repo_path, local_path in files:
    if local_path.exists():
        print(f"  Already present: {local_path}")
        continue
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {repo_path} from {repo_id} ...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=repo_path,
        repo_type="dataset",
        local_dir=".",
    )
    print(f"    saved to {downloaded}")

print("Artifacts ready.")
PYEOF
fi

# Fallback: if artifacts still missing, rebuild locally from Input_Data/.
if [ ! -f "chunked_papers.json" ]; then
    echo "chunked_papers.json not found — running chunking pipeline..."
    python3 chunking.py
fi

if [ ! -f "faiss_index/astrophysics.index" ]; then
    echo "FAISS index not found — running embedding pipeline (this will take a while on CPU)..."
    python3 create_embeddings.py
fi

echo "Starting MCP server..."
python3 mcp_server.py
