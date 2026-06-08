"""
Downloads astrophysics papers from arXiv and saves them as text files.

Usage:
    python3 download_papers.py

Output: Input_Data/<arxiv_id>.txt  (one file per paper)
"""

import time
import requests
import arxiv
from pathlib import Path
from pypdf import PdfReader
from io import BytesIO

OUTPUT_DIR = Path("Input_Data")
TARGET = 1400

CATEGORIES = [
    "astro-ph.GA",  # Galaxies
    "astro-ph.CO",  # Cosmology
    "astro-ph.HE",  # High Energy Astrophysics
    "astro-ph.SR",  # Solar and Stellar
    "astro-ph.EP",  # Earth and Planetary
    "astro-ph.IM",  # Instrumentation and Methods
]

client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)

existing_ids = {f.stem for f in OUTPUT_DIR.glob("*.txt")}
print(f"Already have {len(existing_ids)} papers. Downloading {TARGET} more...\n")

downloaded = 0

for category in CATEGORIES:
    if downloaded >= TARGET:
        break

    print(f"--- Searching {category} ---")

    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=400,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    for paper in client.results(search):
        if downloaded >= TARGET:
            break

        safe_id = paper.get_short_id().replace("/", "_")

        if safe_id in existing_ids:
            continue

        try:
            # Download PDF into memory — no temp file needed
            response = requests.get(paper.pdf_url, timeout=30)
            response.raise_for_status()

            reader = PdfReader(BytesIO(response.content))
            pages = [page.extract_text() or "" for page in reader.pages]
            full_text = "\n\n".join(pages)

            if len(full_text.strip()) < 500:
                print(f"  SKIP {safe_id}: too little text extracted")
                continue

            txt_path = OUTPUT_DIR / f"{safe_id}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Title: {paper.title}\n")
                f.write(f"Authors: {', '.join(str(a) for a in paper.authors)}\n")
                f.write(f"Published: {paper.published.date()}\n")
                f.write(f"Categories: {', '.join(paper.categories)}\n\n")
                f.write(full_text)

            existing_ids.add(safe_id)
            downloaded += 1
            print(f"  [{downloaded}/{TARGET}] {safe_id}: {paper.title[:70]}")

        except Exception as e:
            print(f"  FAIL {safe_id}: {e}")

        time.sleep(1)

print(f"\nDone. Downloaded {downloaded} new papers.")
print(f"Total papers in Input_Data: {len(existing_ids)}")
print("\nNext steps:")
print("  python3 chunking.py")
print("  python3 create_embeddings.py")
