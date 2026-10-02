from pathlib import Path

DATA_FOLDER = Path(r"C:\Users\srira\Documents\local-RAG\test-data")

CHUNK_SIZE = 300


def create_chunks(text):
    # Split the document into paragraphs
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:

        # If adding this paragraph stays within our target size
        if len(current_chunk) + len(paragraph) <= CHUNK_SIZE:

            if current_chunk:
                current_chunk += "\n\n"

            current_chunk += paragraph

        else:
            # Save the current chunk
            if current_chunk:
                chunks.append(current_chunk)

            # Start a new chunk
            current_chunk = paragraph

    # Save the final chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


for file_path in DATA_FOLDER.rglob("*.txt"):

    print(f"\n--- FILE: {file_path.name} ---")

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    chunks = create_chunks(content)

    for number, chunk in enumerate(chunks, start=1):

        print(f"\n--- CHUNK {number} ---")
        print(chunk)