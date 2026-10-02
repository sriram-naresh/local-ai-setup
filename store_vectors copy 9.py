from pathlib import Path
import hashlib

from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
from docx import Document


# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_FOLDERS = [
    Path(r"C:\Users\srira\Documents\local-RAG\test-data"),
]

COLLECTION_NAME = "my_documents"

# Keep False.
# We already migrated the collection.
REBUILD_COLLECTION = False


# ==================================================
# 2. SUPPORTED FILE TYPES
# ==================================================

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
}


# ==================================================
# 3. READ TEXT FILE
# ==================================================

def read_txt(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# ==================================================
# 4. READ PDF FILE
# ==================================================

def read_pdf(file_path):

    reader = PdfReader(
        str(file_path)
    )

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:

            pages.append(
                text
            )

    return "\n\n".join(
        pages
    )


# ==================================================
# 5. READ DOCX FILE
# ==================================================

def read_docx(file_path):

    document = Document(
        str(file_path)
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraphs.append(
                text
            )

    return "\n\n".join(
        paragraphs
    )


# ==================================================
# 6. GENERIC FILE READER
# ==================================================

def read_file(file_path):

    extension = (
        file_path.suffix.lower()
    )


    if extension == ".txt":

        return read_txt(
            file_path
        )


    elif extension == ".pdf":

        return read_pdf(
            file_path
        )


    elif extension == ".docx":

        return read_docx(
            file_path
        )


    return None


# ==================================================
# 7. CREATE CHUNKS
# ==================================================

def create_chunks(text):

    paragraphs = text.split(
        "\n\n"
    )

    chunks = [

        paragraph.strip()

        for paragraph in paragraphs

        if paragraph.strip()
    ]

    return chunks


# ==================================================
# 8. CALCULATE FILE HASH
# ==================================================

def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()


    with open(
        file_path,
        "rb"
    ) as file:

        while chunk := file.read(4096):

            sha256.update(
                chunk
            )


    return sha256.hexdigest()


# ==================================================
# 9. GET UNIQUE SOURCE PATH
# ==================================================

def get_source_path(file_path):

    return str(
        file_path.resolve()
    ).replace(
        "\\",
        "/"
    )


# ==================================================
# 10. CREATE UNIQUE CHUNK ID
# ==================================================

def create_chunk_id(
    source,
    chunk_number
):

    source_hash = hashlib.sha256(

        source.encode(
            "utf-8"
        )

    ).hexdigest()


    return (
        f"{source_hash}_{chunk_number}"
    )


# ==================================================
# 11. LOAD EMBEDDING MODEL
# ==================================================

print(
    "Loading embedding model..."
)


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# 12. CONNECT TO CHROMADB
# ==================================================

client = chromadb.PersistentClient(

    path="./chroma_db"
)


# ==================================================
# 13. CREATE / LOAD COLLECTION
# ==================================================

if REBUILD_COLLECTION:

    print(
        "\nRebuilding collection..."
    )


    try:

        client.delete_collection(
            COLLECTION_NAME
        )

        print(
            "Old collection deleted."
        )


    except Exception:

        print(
            "No existing collection found."
        )


    collection = client.create_collection(

        COLLECTION_NAME
    )


    print(
        "New collection created."
    )


else:

    try:

        collection = client.get_collection(

            COLLECTION_NAME
        )


        print(
            "Existing collection loaded."
        )


    except Exception:

        collection = client.create_collection(

            COLLECTION_NAME
        )


        print(
            "New collection created."
        )


# ==================================================
# 14. GET EXISTING FILE INFORMATION
# ==================================================

existing_data = collection.get(

    include=[
        "metadatas"
    ]
)


existing_files = {}


for metadata in existing_data[
    "metadatas"
]:

    source = metadata.get(
        "source"
    )


    file_hash = metadata.get(
        "file_hash"
    )


    if source and file_hash:

        existing_files[
            source
        ] = file_hash


# ==================================================
# 15. TRACK CURRENT FILES
# ==================================================

current_files = set()


# ==================================================
# 16. SCAN CONFIGURED FOLDERS
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
    # Scan recursively
    # --------------------------------------------------

    for file_path in data_folder.rglob("*"):


        # --------------------------------------------------
        # Ignore directories
        # --------------------------------------------------

        if not file_path.is_file():

            continue


        # --------------------------------------------------
        # Get extension
        # --------------------------------------------------

        extension = (

            file_path.suffix.lower()

        )


        # --------------------------------------------------
        # Ignore unsupported files
        # --------------------------------------------------

        if extension not in SUPPORTED_EXTENSIONS:

            continue


        # --------------------------------------------------
        # Get unique source
        # --------------------------------------------------

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
        # 17. CALCULATE CURRENT HASH
        # ==================================================

        current_hash = calculate_file_hash(
            file_path
        )


        # ==================================================
        # 18. NEW FILE
        # ==================================================

        if source not in existing_files:

            print(
                "Status: NEW"
            )


            # ----------------------------------------------
            # Read file
            # ----------------------------------------------

            try:

                content = read_file(
                    file_path
                )

            except Exception as error:

                print(
                    f"ERROR reading file: {error}"
                )

                continue


            # ----------------------------------------------
            # Empty file
            # ----------------------------------------------

            if not content:

                print(
                    "WARNING: No text extracted."
                )

                continue


            # ----------------------------------------------
            # Create chunks
            # ----------------------------------------------

            chunks = create_chunks(
                content
            )


            # ----------------------------------------------
            # Generate embeddings
            # ----------------------------------------------

            for number, chunk in enumerate(

                chunks,

                start=1

            ):


                embedding = model.encode(
                    chunk
                ).tolist()


                # ------------------------------------------
                # Create unique chunk ID
                # ------------------------------------------

                chunk_id = create_chunk_id(

                    source,

                    number
                )


                # ------------------------------------------
                # Store in ChromaDB
                # ------------------------------------------

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

                            "file_hash": current_hash,

                            "file_type": extension

                        }

                    ]

                )


            continue


        # ==================================================
        # 19. EXISTING FILE
        # ==================================================

        old_hash = existing_files[
            source
        ]


        # ==================================================
        # 20. UNCHANGED FILE
        # ==================================================

        if current_hash == old_hash:

            print(
                "Status: UNCHANGED"
            )

            continue


        # ==================================================
        # 21. MODIFIED FILE
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

            include=[
                "metadatas"
            ]

        )


        old_ids = old_chunks[
            "ids"
        ]


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

        try:

            content = read_file(
                file_path
            )

        except Exception as error:

            print(
                f"ERROR reading file: {error}"
            )

            continue


        if not content:

            print(
                "WARNING: No text extracted."
            )

            continue


        # ----------------------------------------------
        # Create chunks
        # ----------------------------------------------

        chunks = create_chunks(
            content
        )


        # ----------------------------------------------
        # Re-index chunks
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

                        "file_hash": current_hash,

                        "file_type": extension

                    }

                ]

            )


# ==================================================
# 22. DETECT DELETED FILES
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

            include=[
                "metadatas"
            ]

        )


        deleted_ids = deleted_chunks[
            "ids"
        ]


        if deleted_ids:

            collection.delete(

                ids=deleted_ids

            )


# ==================================================
# 23. FINAL SUMMARY
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