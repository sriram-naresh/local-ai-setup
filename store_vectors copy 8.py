from pathlib import Path
import hashlib

from sentence_transformers import SentenceTransformer
import chromadb


# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_FOLDERS = [
    Path(r"C:\Users\srira\Documents\local-RAG\test-data"),
]

COLLECTION_NAME = "my_documents"

# IMPORTANT:
# Set this to True ONLY for the migration run.
# After the first successful run, change it to False.
REBUILD_COLLECTION = False


# ==================================================
# 2. CREATE CHUNKS
# ==================================================

def create_chunks(text):

    paragraphs = text.split("\n\n")

    chunks = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    return chunks


# ==================================================
# 3. CALCULATE FILE HASH
# ==================================================

def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while chunk := file.read(4096):

            sha256.update(chunk)

    return sha256.hexdigest()


# ==================================================
# 4. GET UNIQUE FILE PATH
# ==================================================

def get_source_path(file_path):

    return str(
        file_path.resolve()
    ).replace("\\", "/")


# ==================================================
# 5. CREATE UNIQUE CHUNK ID
# ==================================================

def create_chunk_id(source, chunk_number):

    source_hash = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()

    return f"{source_hash}_{chunk_number}"


# ==================================================
# 6. LOAD EMBEDDING MODEL
# ==================================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# 7. CONNECT TO CHROMADB
# ==================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)


# ==================================================
# 8. CREATE / MIGRATE COLLECTION
# ==================================================

if REBUILD_COLLECTION:

    print("\nRebuilding collection...")

    try:

        client.delete_collection(
            COLLECTION_NAME
        )

        print("Old collection deleted.")

    except Exception:

        print("No existing collection found.")

    collection = client.create_collection(
        COLLECTION_NAME
    )

    print("New collection created.")

else:

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


# ==================================================
# 9. GET EXISTING FILE INFORMATION
# ==================================================

existing_data = collection.get(
    include=["metadatas"]
)


existing_files = {}


for metadata in existing_data["metadatas"]:

    source = metadata.get("source")

    file_hash = metadata.get("file_hash")

    if source and file_hash:

        existing_files[source] = file_hash


# ==================================================
# 10. TRACK FILES CURRENTLY ON DISK
# ==================================================

current_files = set()


# ==================================================
# 11. SCAN ALL CONFIGURED FOLDERS
# ==================================================

for data_folder in DATA_FOLDERS:

    print(
        f"\nScanning folder: {data_folder}"
    )


    # --------------------------------------------------
    # Check folder exists
    # --------------------------------------------------

    if not data_folder.exists():

        print(
            f"WARNING: Folder does not exist: {data_folder}"
        )

        continue


    # --------------------------------------------------
    # Find TXT files
    # --------------------------------------------------

    for file_path in data_folder.rglob("*.txt"):

        source = get_source_path(
            file_path
        )

        current_files.add(
            source
        )

        print(
            f"\nChecking: {source}"
        )


        # ==================================================
        # CALCULATE CURRENT HASH
        # ==================================================

        current_hash = calculate_file_hash(
            file_path
        )


        # ==================================================
        # NEW FILE
        # ==================================================

        if source not in existing_files:

            print("Status: NEW")


            # ----------------------------------------------
            # Read file
            # ----------------------------------------------

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                content = file.read()


            # ----------------------------------------------
            # Create chunks
            # ----------------------------------------------

            chunks = create_chunks(
                content
            )


            # ----------------------------------------------
            # Create embeddings and store
            # ----------------------------------------------

            for number, chunk in enumerate(
                chunks,
                start=1
            ):

                embedding = model.encode(
                    chunk
                ).tolist()


                chunk_id = create_chunk_id(
                    source,
                    number
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


        # ==================================================
        # EXISTING FILE
        # ==================================================

        old_hash = existing_files[
            source
        ]


        # ==================================================
        # UNCHANGED FILE
        # ==================================================

        if current_hash == old_hash:

            print(
                "Status: UNCHANGED"
            )

            continue


        # ==================================================
        # MODIFIED FILE
        # ==================================================

        print(
            "Status: MODIFIED"
        )


        # ----------------------------------------------
        # Find old chunks
        # ----------------------------------------------

        old_chunks = collection.get(

            where={
                "source": source
            },

            include=["metadatas"]
        )


        old_ids = old_chunks["ids"]


        # ----------------------------------------------
        # Delete old chunks
        # ----------------------------------------------

        if old_ids:

            collection.delete(
                ids=old_ids
            )


        # ----------------------------------------------
        # Read modified file
        # ----------------------------------------------

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            content = file.read()


        # ----------------------------------------------
        # Create new chunks
        # ----------------------------------------------

        chunks = create_chunks(
            content
        )


        # ----------------------------------------------
        # Store new chunks
        # ----------------------------------------------

        for number, chunk in enumerate(
            chunks,
            start=1
        ):

            embedding = model.encode(
                chunk
            ).tolist()


            chunk_id = create_chunk_id(
                source,
                number
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


# ==================================================
# 12. DETECT DELETED FILES
# ==================================================

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


# ==================================================
# 13. FINAL SUMMARY
# ==================================================

print(
    "\n========================================"
)

print(
    "Incremental ingestion complete."
)

print(
    "Total chunks:",
    collection.count()
)

print(
    "========================================"
)