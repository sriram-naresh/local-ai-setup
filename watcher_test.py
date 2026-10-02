from pathlib import Path
import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


WATCH_FOLDER = Path(
    r"C:\Users\srira\Documents\local-RAG\test-data"
)


class FileChangeHandler(
    FileSystemEventHandler
):

    def on_created(self, event):

        if event.is_directory:
            return

        print(
            f"[CREATED] {event.src_path}"
        )


    def on_modified(self, event):

        if event.is_directory:
            return

        print(
            f"[MODIFIED] {event.src_path}"
        )


    def on_deleted(self, event):

        if event.is_directory:
            return

        print(
            f"[DELETED] {event.src_path}"
        )


    def on_moved(self, event):

        if event.is_directory:
            return

        print(
            f"[MOVED] {event.src_path}"
            f" -> {event.dest_path}"
        )


# ============================================================
# START WATCHER
# ============================================================

if not WATCH_FOLDER.exists():

    print(
        f"Folder does not exist: {WATCH_FOLDER}"
    )

    raise SystemExit(1)


event_handler = FileChangeHandler()

observer = Observer()

observer.schedule(
    event_handler,
    str(WATCH_FOLDER),
    recursive=True
)

observer.start()


print(
    "========================================"
)

print(
    "Folder watcher started."
)

print(
    f"Watching: {WATCH_FOLDER}"
)

print(
    "Make a file change in that folder."
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