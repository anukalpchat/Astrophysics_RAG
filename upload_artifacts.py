"""
One-time script to upload the prebuilt RAG artifacts to Hugging Face Hub.
Run this locally once after building the index.

Usage:
    hf auth login        # paste your HF write token when prompted
    python3 upload_artifacts.py
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

ARTIFACTS = [
    (BASE_DIR / "chunked_papers.json",                "chunked_papers.json"),
    (BASE_DIR / "faiss_index" / "astrophysics.index", "faiss_index/astrophysics.index"),
    (BASE_DIR / "faiss_index" / "metadata.pkl",       "faiss_index/metadata.pkl"),
    (BASE_DIR / "faiss_index" / "config.json",        "faiss_index/config.json"),
]

api = HfApi()

# Auto-detect the logged-in username
user = api.whoami()
username = user["name"]
print(f"Logged in as: {username}")

REPO_ID = f"{username}/astrophysics-rag-artifacts"

# Create the dataset repo if it doesn't exist yet
create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True)
print(f"Repo: {REPO_ID}\n")

for local_path, repo_path in ARTIFACTS:
    if not local_path.exists():
        print(f"  SKIP (not found): {local_path}")
        continue
    size_mb = local_path.stat().st_size / 1e6
    print(f"  Uploading {repo_path} ({size_mb:.0f} MB) ...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"    done.")

print(f"\nAll artifacts uploaded.")
print(f"Set this as an env var on Railway:\n  HF_REPO_ID={REPO_ID}")
