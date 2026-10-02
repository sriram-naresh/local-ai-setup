from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb


DATA_FOLDER = Path(
    r"C:\Users\srira\Documents\local-RAG\test-data"
)


# --------------------------------------------------
# 1. Create chunks
# --------------------------------------------------

def create_chunks(text):

    paragraphs = text.split("\n\n")

    chunks = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    return chunks


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 3. Connect to local ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)


# --------------------------------------------------
# 4. Delete old collection
# --------------------------------------------------

try:

    client.delete_collection(
        "my_documents"
    )

    print("Old collection deleted.")

except Exception:

    print("No existing collection found.")


# --------------------------------------------------
# 5. Create new collection
# --------------------------------------------------

collection = client.create_collection(
    "my_documents"
)


# --------------------------------------------------
# 6. Process every text file
# --------------------------------------------------

for file_path in DATA_FOLDER.rglob("*.txt"):

    print(f"Processing: {file_path.name}")

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        content = file.read()


    chunks = create_chunks(
        content
    )


    # --------------------------------------------------
    # 7. Create embedding for every chunk
    # --------------------------------------------------

    for number, chunk in enumerate(
        chunks,
        start=1
    ):

        embedding = model.encode(
            chunk
        ).tolist()


        chunk_id = (
            f"{file_path.stem}_{number}"
        )


        # --------------------------------------------------
        # 8. Store chunk + embedding + metadata
        # --------------------------------------------------

        collection.add(

            ids=[
                chunk_id
            ],

            documents=[
                chunk
            ],

            embeddings=[
                embedding
            ],

            metadatas=[
                {
                    "source": file_path.name,
                    "chunk": number
                }
            ]
        )


# --------------------------------------------------
# 9. Show final result
# --------------------------------------------------

print("\nDocuments stored in ChromaDB!")

print(
    "Total chunks:",
    collection.count()
)