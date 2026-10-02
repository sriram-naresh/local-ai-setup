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

COLLECTION_NAME = "my_documents"


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
# 6. Get or create collection
# --------------------------------------------------

try:

    collection = client.get_collection(
        COLLECTION_NAME
    )

    print("Existing collection loaded.")

except Exception:

    collection = client.create_collection(
        COLLECTION_NAME
    )

    print("New collection created.")


# --------------------------------------------------
# 7. Get existing files from ChromaDB
# --------------------------------------------------

existing_data = collection.get(
    include=["metadatas"]
)


existing_files = {}


for metadata in existing_data["metadatas"]:

    source = metadata.get("source")

    file_hash = metadata.get("file_hash")

    if source and file_hash:

        existing_files[source] = file_hash


# --------------------------------------------------
# 8. Track files currently on disk
# --------------------------------------------------

current_files = set()


# --------------------------------------------------
# 9. Process files
# --------------------------------------------------

for file_path in DATA_FOLDER.rglob("*.txt"):

    file_name = file_path.name

    current_files.add(file_name)

    print(
        f"\nChecking: {file_name}"
    )


    # --------------------------------------------------
    # Calculate current hash
    # --------------------------------------------------

    current_hash = calculate_file_hash(
        file_path
    )


    # --------------------------------------------------
    # New file
    # --------------------------------------------------

    if file_name not in existing_files:

        print("Status: NEW")

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            content = file.read()


        chunks = create_chunks(
            content
        )


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
                        "source": file_name,

                        "chunk": number,

                        "file_hash": current_hash
                    }

                ]
            )


        continue


    # --------------------------------------------------
    # Existing file
    # --------------------------------------------------

    old_hash = existing_files[
        file_name
    ]


    # --------------------------------------------------
    # Unchanged file
    # --------------------------------------------------

    if current_hash == old_hash:

        print("Status: UNCHANGED")

        continue


    # --------------------------------------------------
    # Modified file
    # --------------------------------------------------

    print("Status: MODIFIED")


    # --------------------------------------------------
    # Delete old chunks
    # --------------------------------------------------

    old_chunks = collection.get(

        where={
            "source": file_name
        },

        include=["metadatas"]
    )


    old_ids = old_chunks["ids"]


    if old_ids:

        collection.delete(
            ids=old_ids
        )


    # --------------------------------------------------
    # Read modified file
    # --------------------------------------------------

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
    # Re-index modified file
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
                    "source": file_name,

                    "chunk": number,

                    "file_hash": current_hash
                }

            ]
        )


# --------------------------------------------------
# 10. Detect deleted files
# --------------------------------------------------

for file_name in existing_files:

    if file_name not in current_files:

        print(
            f"\nStatus: DELETED - {file_name}"
        )


        deleted_chunks = collection.get(

            where={
                "source": file_name
            },

            include=["metadatas"]
        )


        deleted_ids = deleted_chunks["ids"]


        if deleted_ids:

            collection.delete(
                ids=deleted_ids
            )


# --------------------------------------------------
# 11. Final summary
# --------------------------------------------------

print(
    "\n--------------------------------"
)

print(
    "Incremental ingestion complete."
)

print(
    "Total chunks:",
    collection.count()
)

print(
    "--------------------------------"
)