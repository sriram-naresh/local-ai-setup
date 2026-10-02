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
# 4. Get relative file path
# --------------------------------------------------

def get_source_path(file_path):

    return str(
        file_path.relative_to(DATA_FOLDER)
    ).replace("\\", "/")


# --------------------------------------------------
# 5. Load embedding model
# --------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 6. Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)


# --------------------------------------------------
# 7. Get or create collection
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
# 8. Read existing metadata
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
# 9. Track files currently on disk
# --------------------------------------------------

current_files = set()


# --------------------------------------------------
# 10. Process files
# --------------------------------------------------

for file_path in DATA_FOLDER.rglob("*.txt"):

    source = get_source_path(
        file_path
    )

    current_files.add(source)

    print(
        f"\nChecking: {source}"
    )


    # --------------------------------------------------
    # Calculate current hash
    # --------------------------------------------------

    current_hash = calculate_file_hash(
        file_path
    )


    # --------------------------------------------------
    # NEW FILE
    # --------------------------------------------------

    if source not in existing_files:

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
                f"{hashlib.sha256(source.encode()).hexdigest()}_{number}"
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
                        "source": source,

                        "chunk": number,

                        "file_hash": current_hash
                    }

                ]
            )


        continue


    # --------------------------------------------------
    # EXISTING FILE
    # --------------------------------------------------

    old_hash = existing_files[
        source
    ]


    # --------------------------------------------------
    # UNCHANGED
    # --------------------------------------------------

    if current_hash == old_hash:

        print("Status: UNCHANGED")

        continue


    # --------------------------------------------------
    # MODIFIED
    # --------------------------------------------------

    print("Status: MODIFIED")


    # Delete old chunks

    old_chunks = collection.get(

        where={
            "source": source
        },

        include=["metadatas"]
    )


    old_ids = old_chunks["ids"]


    if old_ids:

        collection.delete(
            ids=old_ids
        )


    # Read modified file

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        content = file.read()


    chunks = create_chunks(
        content
    )


    # Re-index modified file

    for number, chunk in enumerate(
        chunks,
        start=1
    ):

        embedding = model.encode(
            chunk
        ).tolist()


        chunk_id = (
            f"{hashlib.sha256(source.encode()).hexdigest()}_{number}"
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
                    "source": source,

                    "chunk": number,

                    "file_hash": current_hash
                }

            ]
        )


# --------------------------------------------------
# 11. Detect deleted files
# --------------------------------------------------

for source in existing_files:

    if source not in current_files:

        print(
            f"\nStatus: DELETED - {source}"
        )


        deleted_chunks = collection.get(

            where={
                "source": source
            },

            include=["metadatas"]
        )


        deleted_ids = deleted_chunks["ids"]


        if deleted_ids:

            collection.delete(
                ids=deleted_ids
            )


# --------------------------------------------------
# 12. Summary
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