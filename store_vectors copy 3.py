from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

DATA_FOLDER = Path(r"C:\Users\srira\Documents\local-RAG\test-data")

CHUNK_SIZE = 300


def create_chunks(text):

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        if len(current_chunk) + len(paragraph) <= CHUNK_SIZE:

            if current_chunk:
                current_chunk += "\n\n"

            current_chunk += paragraph

        else:

            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to local ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")


# Delete old collection
try:
    client.delete_collection("my_documents")
    print("Old collection deleted.")
except Exception:
    print("No existing collection found.")


# Create new collection
collection = client.create_collection("my_documents")


# Process files
for file_path in DATA_FOLDER.rglob("*.txt"):

    print(f"Processing: {file_path.name}")

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    chunks = create_chunks(content)

    for number, chunk in enumerate(chunks, start=1):

        embedding = model.encode(chunk).tolist()

        chunk_id = f"{file_path.stem}_{number}"

        collection.add(
            ids=[chunk_id],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[
                {
                    "source": file_path.name,
                    "chunk": number
                }
            ]
        )


print("\nDocuments stored in ChromaDB!")
print("Total chunks:", collection.count())