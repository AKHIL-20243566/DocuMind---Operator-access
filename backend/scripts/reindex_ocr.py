"""Re-indexing utility — migrate from Tesseract OCR to PaddleOCR.

Usage
-----
Run from the project root (or backend/) after installing paddleocr/paddlepaddle:

    python backend/scripts/reindex_ocr.py [--vector-data PATH] [--purge]

What it does
------------
1. Clears stale OCR cache (vector_data/ocr_cache.json — Tesseract results).
2. Scans vector_data/documents.json for OCR-tagged chunks (chunk_id contains "_ocr_").
3. Reports which source files were originally processed via OCR.
4. Prints the curl commands you need to run to re-ingest those files via the API
   so they are re-processed with PaddleOCR and re-indexed in FAISS.

If --purge is given, it also removes all OCR-tagged chunks from documents.json
AND rebuilds the FAISS index from the remaining non-OCR chunks.  You will still
need to re-upload the OCR files to get them back into the index.

NOTE: PaddleOCR will download model weights (~50 MB) on the very first OCR call.
Ensure the host has internet access on first run, or pre-download the models:
    python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> list | dict | None:
    if not path.exists():
        print(f"[WARN] {path} not found — skipping.")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERROR] Could not parse {path}: {exc}")
        return None


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DocuMind OCR re-indexing utility")
    parser.add_argument(
        "--vector-data",
        default="vector_data",
        help="Path to the vector_data directory (default: vector_data)",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help=(
            "Remove all OCR-tagged chunks from documents.json and rebuild FAISS. "
            "You must re-upload the affected files afterwards."
        ),
    )
    args = parser.parse_args()

    vd = Path(args.vector_data)

    # -----------------------------------------------------------------------
    # Step 1: Clear stale OCR cache
    # -----------------------------------------------------------------------
    cache_path = vd / "ocr_cache.json"
    if cache_path.exists():
        size = cache_path.stat().st_size
        cache_path.unlink()
        print(f"[OK] Cleared stale OCR cache: {cache_path} ({size:,} bytes removed)")
    else:
        print(f"[INFO] No existing OCR cache at {cache_path}")

    # -----------------------------------------------------------------------
    # Step 2: Analyse documents.json
    # -----------------------------------------------------------------------
    docs_path = vd / "documents.json"
    documents = load_json(docs_path)
    if documents is None:
        print("[INFO] No documents.json found — nothing to re-index.")
        sys.exit(0)

    if not isinstance(documents, list):
        print(f"[ERROR] documents.json has unexpected format (expected list). Aborting.")
        sys.exit(1)

    total_chunks   = len(documents)
    ocr_chunks     = [d for d in documents if "_ocr_" in d.get("chunk_id", "")]
    non_ocr_chunks = [d for d in documents if "_ocr_" not in d.get("chunk_id", "")]

    # Collect unique source filenames that have OCR chunks
    ocr_sources: dict[str, int] = {}
    for chunk in ocr_chunks:
        src = chunk.get("source", "<unknown>")
        ocr_sources[src] = ocr_sources.get(src, 0) + 1

    print()
    print("=" * 60)
    print("  DocuMind OCR Migration Report")
    print("=" * 60)
    print(f"  Total chunks in index : {total_chunks}")
    print(f"  OCR-tagged chunks     : {len(ocr_chunks)}")
    print(f"  Non-OCR chunks        : {len(non_ocr_chunks)}")
    print(f"  Files needing re-ingest: {len(ocr_sources)}")
    print()

    if not ocr_sources:
        print("[OK] No OCR-tagged chunks found. Index is clean — nothing to do.")
        sys.exit(0)

    print("Files originally processed via OCR (must be re-uploaded):")
    for src, count in sorted(ocr_sources.items()):
        print(f"  • {src}  ({count} chunks)")

    # -----------------------------------------------------------------------
    # Step 3: Print re-ingest commands
    # -----------------------------------------------------------------------
    print()
    print("Re-ingest commands (replace localhost:8000 with your API URL):")
    print("-" * 60)
    for src in sorted(ocr_sources.keys()):
        print(
            f'  curl -X DELETE "http://localhost:8000/documents/{src}"'
        )
    print()
    for src in sorted(ocr_sources.keys()):
        print(
            f'  curl -X POST "http://localhost:8000/upload" \\'
            f'\n    -F "file=@{src}"'
        )

    # -----------------------------------------------------------------------
    # Step 4 (optional): Purge OCR chunks and rebuild FAISS
    # -----------------------------------------------------------------------
    if args.purge:
        print()
        print("[PURGE] Removing OCR-tagged chunks from documents.json ...")
        save_json(docs_path, non_ocr_chunks)
        print(f"[OK] documents.json updated: {len(non_ocr_chunks)} chunks retained, "
              f"{len(ocr_chunks)} OCR chunks removed.")

        # Rebuild FAISS from scratch using remaining embeddings
        print("[PURGE] Rebuilding FAISS index from non-OCR chunks ...")
        try:
            # Ensure backend directory is on the path
            import os
            backend_dir = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(backend_dir))
            os.chdir(backend_dir)

            import numpy as np
            from vector_store import _rebuild_faiss, save_index  # type: ignore

            emb_path = vd / "embeddings.npy"
            if not emb_path.exists():
                print("[WARN] embeddings.npy not found — FAISS rebuild skipped. "
                      "Restart the backend; it will rebuild on next request.")
            else:
                all_embs = np.load(str(emb_path))
                # Keep only embeddings whose parallel document is a non-OCR chunk
                original_docs = load_json(docs_path) or []
                # After the save above, docs_path already has non_ocr_chunks.
                # But all_embs still has ALL embeddings (before purge).
                # We need the original full document list to find the correct indices.
                # Re-load the backup list we built before saving.
                keep_indices = [
                    i for i, d in enumerate(documents)
                    if "_ocr_" not in d.get("chunk_id", "")
                ]
                if keep_indices:
                    filtered_embs = all_embs[np.array(keep_indices)]
                    np.save(str(emb_path), filtered_embs)
                    _rebuild_faiss(filtered_embs)
                    save_index()
                    print(f"[OK] FAISS rebuilt with {len(keep_indices)} embeddings.")
                else:
                    print("[WARN] No non-OCR embeddings to keep — FAISS index will be empty.")

        except Exception as exc:
            print(f"[ERROR] FAISS rebuild failed: {exc}")
            print("        The documents.json has been cleaned but FAISS may be stale.")
            print("        Restart the backend — it will reload from the cleaned documents.json.")
    else:
        print()
        print("Run with --purge to remove stale OCR chunks from the index automatically.")

    print()
    print("Done. After re-uploading files, the new PaddleOCR results will be indexed.")


if __name__ == "__main__":
    main()
