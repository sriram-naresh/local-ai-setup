from pathlib import Path

DATA_FOLDER = Path(r"C:\Users\srira\Documents\local-RAG\test-data")

CHUNK_SIZE = 500


for file_path in DATA_FOLDER.rglob("*.txt"):
    print(f"\n--- FILE: {file_path.name} ---")

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    chunks = [
        content[i:i + CHUNK_SIZE]
        for i in range(0, len(content), CHUNK_SIZE)
    ]

    for number, chunk in enumerate(chunks, start=1):
        print(f"\n--- CHUNK {number} ---")
        print(chunk)