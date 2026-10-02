from pathlib import Path

import ingestion


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FOLDERS = [
    Path(r"C:\Users\srira\Documents"),
    Path(r"C:\Users\srira\Downloads"),
    Path(r"C:\Users\srira\Desktop"),
]

EXCLUDED_DIRS = {
    ".git",
    ".svn",
    ".hg",
    "venv",
    ".venv",
    "env",
    ".env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "chroma_db",
    "$Recycle.Bin",
}


# ============================================================
# SHOULD PROCESS FILE?
# ============================================================

def should_process(path):

    path = Path(path)

    # Must be a file
    if not path.is_file():
        return False

    # Only supported extensions
    if path.suffix.lower() not in ingestion.SUPPORTED_EXTENSIONS:
        return False

    # Skip excluded directories
    for part in path.parts:

        if part in EXCLUDED_DIRS:
            return False

    # Skip temporary Microsoft Office files
    if path.name.startswith("~$"):
        return False

    return True


# ============================================================
# FIND SUPPORTED FILES
# ============================================================

def find_files():

    files = []

    for folder in DATA_FOLDERS:

        if not folder.exists():

            print(
                f"WARNING: Folder does not exist: {folder}"
            )

            continue

        print(
            f"Scanning: {folder}"
        )

        try:

            for path in folder.rglob("*"):

                if should_process(path):

                    files.append(path)

        except Exception as e:

            print(
                f"ERROR scanning {folder}: {e}"
            )

    return files


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("              RAG CHROMA REBUILD")
    print("=" * 60)

    print()
    print(
        "This will completely rebuild your Chroma database."
    )

    print()
    print(
        "Your original Documents / Downloads files"
    )

    print(
        "will NOT be deleted."
    )

    print()
    print(
        f"Chroma path: {ingestion.CHROMA_PATH}"
    )

    print(
        f"Collection:  {ingestion.COLLECTION_NAME}"
    )

    print()

    # ========================================================
    # CONFIRMATION
    # ========================================================

    confirmation = input(
        "Type REBUILD to continue: "
    ).strip()

    if confirmation != "REBUILD":

        print()
        print(
            "Rebuild cancelled."
        )

        return

    print()
    print(
        "Preparing Chroma database..."
    )

    # `ingestion` has already opened this SQLite database at
    # import time. Removing its directory here therefore locks on
    # Windows (and can invalidate open clients elsewhere). Clear the
    # collection through Chroma and keep using that existing client.
    collection = ingestion.collection

    try:
        existing = collection.get(include=[])
        existing_ids = existing.get("ids", [])

        if existing_ids:
            print(f"Removing {len(existing_ids)} existing Chroma chunks...")
            collection.delete(ids=existing_ids)
        else:
            print("Chroma collection is already empty.")

    except Exception as e:
        print()
        print("ERROR: Could not clear the Chroma collection.")
        print(f"Reason: {e}")
        print("Close chat_rag_v2.py and rag_watcher.py, then retry.")
        return

    # ========================================================
    # FIND FILES
    # ========================================================

    print()
    print("=" * 60)
    print("SCANNING DOCUMENTS")
    print("=" * 60)

    files = find_files()

    print()
    print(
        f"Found {len(files)} supported files."
    )

    if not files:

        print()
        print(
            "No supported files found."
        )

        return

    # ========================================================
    # INGEST FILES
    # ========================================================

    print()
    print("=" * 60)
    print("STARTING REBUILD")
    print("=" * 60)

    successful = 0
    failed = 0

    for index, file_path in enumerate(
        files,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(files)}] "
            f"{file_path.name}"
        )

        try:

            success = ingestion.ingest_file(
                file_path
            )

            if success:

                successful += 1

            else:

                failed += 1

        except Exception as e:

            print()
            print(
                f"FAILED: {file_path}"
            )

            print(
                f"Reason: {e}"
            )

            failed += 1

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    total_chunks = collection.count()

    print()
    print("=" * 60)
    print("              REBUILD COMPLETE")
    print("=" * 60)

    print(
        f"Files found:        {len(files)}"
    )

    print(
        f"Successfully read:  {successful}"
    )

    print(
        f"Failed:             {failed}"
    )

    print(
        f"Chroma chunks:      {total_chunks}"
    )

    print("=" * 60)

    print()
    print(
        "Your Chroma database has been rebuilt."
    )

    print(
        "Source metadata should now be available."
    )

    print()
    print(
        "Next test:"
    )

    print(
        "  python chat_rag_v2.py"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
