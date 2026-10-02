from pathlib import Path
import hashlib

from sentence_transformers import SentenceTransformer
import chromadb


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_FOLDER = Path(
    r"C:\Users\srira\Documents\local-RAG\test-data"
)


# --------------------------------------------------
# 2. Create chunks
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
# 3. Calculate file hash
# --------------------------------------------------

def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while chunk := file.read(4096):

            sha256.update(chunk)

    return sha256.hexdigest()


# --------------------------------------------------
# 4. Load embedding model
# --------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 5. Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)


# --------------------------------------------------
# 6. Delete old collection
# --------------------------------------------------

try:

    client.delete_collection(
        "my_documents"
    )

    print("Old collection deleted.")

except Exception:

    print("No existing collection found.")


# --------------------------------------------------
# 7. Create collection
# --------------------------------------------------

collection = client.create_collection(
    "my_documents"
)


# --------------------------------------------------
# 8. Process files
# --------------------------------------------------

for file_path in DATA_FOLDER.rglob("*.txt"):

    print(
        f"Processing: {file_path.name}"
    )


    # --------------------------------------------------
    # Calculate hash
    # --------------------------------------------------

    file_hash = calculate_file_hash(
        file_path
    )


    # --------------------------------------------------
    # Read file
    # --------------------------------------------------

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        content = file.read()


    # --------------------------------------------------
    # Create chunks
    # --------------------------------------------------

    chunks = create_chunks(
        content
    )


    # --------------------------------------------------
    # Store chunks
    # --------------------------------------------------

    for number, chunk in enumerate(
        chunks,
        start=1
    ):

        # Generate embedding

        embedding = model.encode(
            chunk
        ).tolist()


        # Unique chunk ID

        chunk_id = (
            f"{file_path.stem}_{number}"
        )


        # Store in ChromaDB

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

                    "chunk": number,

                    "file_hash": file_hash
                }

            ]
        )


# --------------------------------------------------
# 9. Summary
# --------------------------------------------------

print(
    "\nDocuments stored in ChromaDB!"
)

print(
    "Total chunks:",
    collection.count()
)