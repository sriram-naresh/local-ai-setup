from pathlib import Path

DATA_FOLDER = Path(r"C:\Users\srira\Documents\local-RAG\test-data")

for file_path in DATA_FOLDER.rglob("*.txt"):
    print(f"\n--- FILE: {file_path.name} ---")

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    print(content)