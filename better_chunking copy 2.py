from pathlib import Path

DATA_FOLDER = Path(r"C:\Users\srira\Documents\local-RAG\test-data")


def create_chunks(text):

    # Each paragraph becomes one chunk
    paragraphs = text.split("\n\n")

    chunks = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    return chunks


for file_path in DATA_FOLDER.rglob("*.txt"):

    print(f"\n--- FILE: {file_path.name} ---")

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    chunks = create_chunks(content)

    for number, chunk in enumerate(chunks, start=1):

        print(f"\n--- CHUNK {number} ---")
        print(chunk)