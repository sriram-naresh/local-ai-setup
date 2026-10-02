from pathlib import Path
import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ingestion import (
    ingest_file,
    delete_file,
    SUPPORTED_EXTENSIONS,
)


# ============================================================
# FOLDERS TO WATCH
# ============================================================

WATCH_FOLDERS = [
    Path(r"C:\Users\srira\Documents"),
    Path(r"C:\Users\srira\Downloads"),
    Path(r"C:\Users\srira\Desktop"),
]


# ============================================================
# DIRECTORIES TO IGNORE
# ============================================================

EXCLUDED_DIRECTORIES = {
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
# FILE CHECK
# ============================================================

def should_process(file_path):

    path = Path(file_path)


    # Must be a file

    if not path.is_file():

        return False


    # Check extension

    if (
        path.suffix.lower()
        not in SUPPORTED_EXTENSIONS
    ):

        return False


    # Check excluded directories

    for part in path.parts:

        if part in EXCLUDED_DIRECTORIES:

            return False


    return True


# ============================================================
# EVENT HANDLER
# ============================================================

class RAGFileHandler(
    FileSystemEventHandler
):


    def on_created(self, event):

        if event.is_directory:

            return


        if not should_process(
            event.src_path
        ):

            return


        print(
            f"\n[CREATED] {event.src_path}"
        )


        # Small delay because some applications
        # create the file before they finish writing it.

        time.sleep(2)


        ingest_file(
            event.src_path
        )


    def on_modified(self, event):

        if event.is_directory:

            return


        if not should_process(
            event.src_path
        ):

            return


        print(
            f"\n[MODIFIED] {event.src_path}"
        )


        # Small delay to allow the application
        # to finish writing the file.

        time.sleep(2)


        ingest_file(
            event.src_path
        )


    def on_deleted(self, event):

        if event.is_directory:

            return


        if not should_process(
            event.src_path
        ):

            return


        print(
            f"\n[DELETED] {event.src_path}"
        )


        delete_file(
            event.src_path
        )


    def on_moved(self, event):

        if event.is_directory:

            return


        # Treat move as delete + create

        if should_process(
            event.src_path
        ):

            print(
                f"\n[MOVED FROM] "
                f"{event.src_path}"
            )

            delete_file(
                event.src_path
            )


        if should_process(
            event.dest_path
        ):

            print(
                f"\n[MOVED TO] "
                f"{event.dest_path}"
            )

            time.sleep(2)

            ingest_file(
                event.dest_path
            )


# ============================================================
# START WATCHER
# ============================================================

def main():

    handler = RAGFileHandler()

    observer = Observer()


    watched_count = 0


    for folder in WATCH_FOLDERS:

        if not folder.exists():

            print(
                f"WARNING: Folder does not exist: "
                f"{folder}"
            )

            continue


        observer.schedule(
            handler,
            str(folder),
            recursive=True
        )


        watched_count += 1


        print(
            f"Watching: {folder}"
        )


    if watched_count == 0:

        print(
            "No valid folders to watch."
        )

        return


    observer.start()


    print(
        "\n========================================"
    )

    print(
        "       LOCAL RAG WATCHER"
    )

    print(
        "========================================"
    )

    print(
        "Watcher is running."
    )

    print(
        "New/modified documents will be indexed."
    )

    print(
        "Deleted documents will be removed."
    )

    print(
        "Press Ctrl+C to stop."
    )

    print(
        "========================================"
    )


    try:

        while True:

            time.sleep(1)


    except KeyboardInterrupt:

        print(
            "\nStopping watcher..."
        )

        observer.stop()


    observer.join()


    print(
        "Watcher stopped."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()