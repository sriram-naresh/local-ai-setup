from pathlib import Path
import time
import threading

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ingestion import (
    ingest_file,
    delete_file,
    SUPPORTED_EXTENSIONS,
)


# ============================================================
# CONFIGURATION
# ============================================================

WATCH_FOLDERS = [
    Path(r"C:\Users\srira\Documents"),
    Path(r"C:\Users\srira\Downloads"),
    Path(r"C:\Users\srira\Desktop"),
]

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


# Wait this long after the LAST file event
# before ingesting the file.
DEBOUNCE_SECONDS = 3


# ============================================================
# FILE CHECK
# ============================================================

def should_process(file_path):

    path = Path(file_path)


    if not path.is_file():

        return False


    if (
        path.suffix.lower()
        not in SUPPORTED_EXTENSIONS
    ):

        return False


    for part in path.parts:

        if part in EXCLUDED_DIRECTORIES:

            return False


    return True


# ============================================================
# DEBOUNCE MANAGER
# ============================================================

class DebounceManager:

    def __init__(self):

        self.timers = {}

        self.lock = threading.Lock()


    def schedule(self, file_path):

        file_path = str(
            Path(file_path).resolve()
        )


        with self.lock:

            # Cancel previous timer for this file

            if file_path in self.timers:

                self.timers[
                    file_path
                ].cancel()


            # Create new timer

            timer = threading.Timer(
                DEBOUNCE_SECONDS,
                self._process,
                args=(file_path,)
            )


            self.timers[
                file_path
            ] = timer


            timer.start()


    def _process(self, file_path):

        with self.lock:

            self.timers.pop(
                file_path,
                None
            )


        path = Path(
            file_path
        )


        # File may have been deleted
        # while the timer was waiting.

        if not path.exists():

            return


        if not should_process(path):

            return


        print(
            f"\n[INDEXING] {path}"
        )


        try:

            ingest_file(path)

        except Exception as e:

            print(
                f"Watcher ingestion error: {e}"
            )


    def cancel(self, file_path):

        file_path = str(
            Path(file_path).resolve()
        )


        with self.lock:

            timer = self.timers.pop(
                file_path,
                None
            )


            if timer:

                timer.cancel()


# ============================================================
# EVENT HANDLER
# ============================================================

class RAGFileHandler(
    FileSystemEventHandler
):

    def __init__(self):

        super().__init__()

        self.debounce = (
            DebounceManager()
        )


    # --------------------------------------------------------
    # CREATED
    # --------------------------------------------------------

    def on_created(self, event):

        if event.is_directory:

            return


        if not should_process(
            event.src_path
        ):

            return


        print(
            f"\n[CREATED] "
            f"{event.src_path}"
        )


        self.debounce.schedule(
            event.src_path
        )


    # --------------------------------------------------------
    # MODIFIED
    # --------------------------------------------------------

    def on_modified(self, event):

        if event.is_directory:

            return


        if not should_process(
            event.src_path
        ):

            return


        print(
            f"\n[MODIFIED] "
            f"{event.src_path}"
        )


        self.debounce.schedule(
            event.src_path
        )


    # --------------------------------------------------------
    # DELETED
    # --------------------------------------------------------

    def on_deleted(self, event):

        if event.is_directory:

            return


        if not should_process(
            event.src_path
        ):

            return


        print(
            f"\n[DELETED] "
            f"{event.src_path}"
        )


        self.debounce.cancel(
            event.src_path
        )


        try:

            delete_file(
                event.src_path
            )

        except Exception as e:

            print(
                f"Delete error: {e}"
            )


    # --------------------------------------------------------
    # MOVED
    # --------------------------------------------------------

    def on_moved(self, event):

        if event.is_directory:

            return


        # Cancel any pending operation
        # for the old path.

        self.debounce.cancel(
            event.src_path
        )


        # Remove old location

        if should_process(
            event.src_path
        ):

            print(
                f"\n[MOVED FROM] "
                f"{event.src_path}"
            )


            try:

                delete_file(
                    event.src_path
                )

            except Exception as e:

                print(
                    f"Delete error: {e}"
                )


        # Index new location

        if should_process(
            event.dest_path
        ):

            print(
                f"\n[MOVED TO] "
                f"{event.dest_path}"
            )


            self.debounce.schedule(
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
        f"Debounce: {DEBOUNCE_SECONDS} seconds"
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
# MAIN
# ============================================================

if __name__ == "__main__":

    main()