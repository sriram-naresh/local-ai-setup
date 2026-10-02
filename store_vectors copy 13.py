from pathlib import Path
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document
from pdf2image import convert_from_path
import pytesseract


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FOLDERS = [
    Path(r"C:\Users\srira\Documents"),
    Path(r"C:\Users\srira\Downloads"),
    Path(r"C:\Users\srira\Desktop"),
]

# IMPORTANT:
# True  = delete existing Chroma collection and rebuild everything
# False = incremental indexing
REBUILD_COLLECTION = False


# Directories that should NOT be scanned
EXCLUDED_DIRECTORIES = {
    ".git",
    ".svn",
    ".hg",
    "venv",
    ".venv",
    "env",
    ".env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "chroma_db",
    "$Recycle.Bin",
}


# Files that should NOT be indexed
EXCLUDED_FILES = {
}


# Supported file types
SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
}


# ============================================================
# CHROMA SETUP
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


client = chromadb.PersistentClient(
    path="./chroma_db"
)


COLLECTION_NAME = "my_documents"


# ============================================================
# CREATE / REBUILD COLLECTION
# ============================================================

if REBUILD_COLLECTION:

    print("\nRebuilding Chroma collection...")

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Existing collection deleted.")
    except Exception:
        print("No existing collection to delete.")


collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

print(f"Using collection: {COLLECTION_NAME}")


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(file_path):
    """
    Calculate SHA256 hash of a file.

    Used to detect whether a file has changed.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            data = f.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# TXT READER
# ============================================================

def read_txt(file_path):

    try:

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception as e:

        print(f"ERROR reading TXT {file_path}: {e}")

        return ""


# ============================================================
# PDF READER + OCR FALLBACK
# ============================================================

def read_pdf(file_path):

    print(f"  Reading PDF: {file_path.name}")

    reader = PdfReader(file_path)

    text = ""

    # --------------------------------------------------------
    # STEP 1: Try normal PDF text extraction
    # --------------------------------------------------------

    for page_number, page in enumerate(reader.pages, start=1):

        try:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        except Exception as e:

            print(
                f"    Warning: Could not extract page "
                f"{page_number}: {e}"
            )

    # --------------------------------------------------------
    # STEP 2: If normal extraction worked, return it
    # --------------------------------------------------------

    if text.strip():

        print("  Normal PDF text extraction successful.")

        return text


    # --------------------------------------------------------
    # STEP 3: No text → OCR fallback
    # --------------------------------------------------------

    print(
        "  No text extracted. Running OCR..."
    )

    try:

        images = convert_from_path(
            file_path,
            dpi=200
        )

    except Exception as e:

        print(
            f"  ERROR converting PDF to images: {e}"
        )

        return ""


    ocr_text = ""


    # --------------------------------------------------------
    # STEP 4: Run Tesseract on every page
    # --------------------------------------------------------

    for page_number, image in enumerate(
        images,
        start=1
    ):

        print(
            f"    OCR page "
            f"{page_number}/{len(images)}"
        )

        try:

            page_text = pytesseract.image_to_string(
                image
            )

            if page_text.strip():

                ocr_text += page_text + "\n"

        except Exception as e:

            print(
                f"    OCR error on page "
                f"{page_number}: {e}"
            )


    # --------------------------------------------------------
    # STEP 5: Return OCR text
    # --------------------------------------------------------

    if ocr_text.strip():

        print("  OCR extraction successful.")

    else:

        print("  WARNING: OCR produced no text.")


    return ocr_text


# ============================================================
# DOCX READER
# SMART CHUNKING
# ============================================================

def read_docx(file_path):

    print(f"  Reading DOCX: {file_path.name}")

    document = Document(file_path)

    sections = []

    current_section = []


    def flush_section():

        if current_section:

            section_text = "\n".join(
                current_section
            ).strip()

            if section_text:

                sections.append(
                    section_text
                )

            current_section.clear()


    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if not text:
            continue


        # ----------------------------------------------------
        # Detect headings
        # ----------------------------------------------------

        style_name = ""

        try:

            style_name = paragraph.style.name

        except Exception:

            pass


        is_heading = (
            style_name.startswith("Heading")
            or text.endswith(":")
        )


        # ----------------------------------------------------
        # If heading:
        # finish previous section and start new section
        # ----------------------------------------------------

        if is_heading:

            flush_section()

            current_section.append(text)

        else:

            current_section.append(text)


    # Add final section

    flush_section()


    return sections


# ============================================================
# TXT / PDF CHUNKING
# ============================================================

def create_text_chunks(text):

    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if paragraph:

            chunks.append(paragraph)


    return chunks


# ============================================================
# FIND ALL FILES
# ============================================================

def find_files():

    files = []


    for data_folder in DATA_FOLDERS:

        if not data_folder.exists():

            print(
                f"WARNING: Folder does not exist: "
                f"{data_folder}"
            )

            continue


        print(
            f"\nScanning: {data_folder}"
        )


        for file_path in data_folder.rglob("*"):

            if not file_path.is_file():
                continue


            # ------------------------------------------------
            # Check excluded directories
            # ------------------------------------------------

            excluded = False

            for part in file_path.parts:

                if part in EXCLUDED_DIRECTORIES:

                    excluded = True

                    break


            if excluded:
                continue


            # ------------------------------------------------
            # Check excluded files
            # ------------------------------------------------

            if file_path.name in EXCLUDED_FILES:

                continue


            # ------------------------------------------------
            # Check extension
            # ------------------------------------------------

            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:

                continue


            files.append(file_path)


    return files


# ============================================================
# GET EXISTING DOCUMENTS FROM CHROMA
# ============================================================

def get_existing_records():

    try:

        results = collection.get(
            include=["metadatas"]
        )

    except Exception as e:

        print(
            f"ERROR reading existing Chroma records: {e}"
        )

        return {}


    existing = {}


    metadatas = results.get(
        "metadatas",
        []
    )


    for metadata in metadatas:

        if not metadata:
            continue


        source = metadata.get("source")

        file_hash = metadata.get("file_hash")


        if source and file_hash:

            existing[source] = file_hash


    return existing


# ============================================================
# PROCESS FILE
# ============================================================

def process_file(file_path):

    extension = file_path.suffix.lower()

    print(
        f"\nProcessing: {file_path}"
    )


    # --------------------------------------------------------
    # Calculate hash
    # --------------------------------------------------------

    file_hash = calculate_file_hash(
        file_path
    )


    # --------------------------------------------------------
    # Read content
    # --------------------------------------------------------

    if extension == ".txt":

        text = read_txt(file_path)

        chunks = create_text_chunks(
            text
        )


    elif extension == ".pdf":

        text = read_pdf(file_path)

        chunks = create_text_chunks(
            text
        )


    elif extension == ".docx":

        chunks = read_docx(file_path)


    else:

        return []


    # --------------------------------------------------------
    # Remove empty chunks
    # --------------------------------------------------------

    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


    if not chunks:

        print(
            "  WARNING: No text/chunks found."
        )

        return []


    print(
        f"  Created {len(chunks)} chunks."
    )


    records = []


    # --------------------------------------------------------
    # Create Chroma records
    # --------------------------------------------------------

    for chunk_number, chunk in enumerate(
        chunks
    ):

        chunk_id = (
            f"{file_hash}_{chunk_number}"
        )


        metadata = {
            "source": str(file_path),
            "chunk": chunk_number,
            "file_hash": file_hash,
            "file_type": extension,
        }


        records.append(
            {
                "id": chunk_id,
                "document": chunk,
                "metadata": metadata,
            }
        )


    return records


# ============================================================
# DELETE OLD CHUNKS FOR FILE
# ============================================================

def delete_file_chunks(source):

    print(
        f"  Removing old chunks: {source}"
    )


    try:

        results = collection.get(
            where={
                "source": source
            },
            include=["metadatas"]
        )

        ids = results.get(
            "ids",
            []
        )


        if ids:

            collection.delete(
                ids=ids
            )


            print(
                f"  Removed {len(ids)} old chunks."
            )


    except Exception as e:

        print(
            f"  ERROR removing old chunks: {e}"
        )


# ============================================================
# MAIN INGESTION
# ============================================================

print("\nFinding files...")

files = find_files()

print(
    f"\nFound {len(files)} supported files."
)


# Existing records in Chroma

existing_records = get_existing_records()


# Track currently existing files

current_files = set()


# Statistics

new_count = 0
modified_count = 0
unchanged_count = 0
deleted_count = 0


# ============================================================
# PROCESS CURRENT FILES
# ============================================================

for file_path in files:

    source = str(file_path)

    current_files.add(source)


    # Calculate current hash

    current_hash = calculate_file_hash(
        file_path
    )


    # --------------------------------------------------------
    # File unchanged
    # --------------------------------------------------------

    if (
        source in existing_records
        and existing_records[source] == current_hash
    ):

        print(
            f"\nUNCHANGED: {file_path.name}"
        )

        unchanged_count += 1

        continue


    # --------------------------------------------------------
    # Modified file
    # --------------------------------------------------------

    if source in existing_records:

        print(
            f"\nMODIFIED: {file_path.name}"
        )

        delete_file_chunks(
            source
        )

        modified_count += 1


    # --------------------------------------------------------
    # New file
    # --------------------------------------------------------

    else:

        print(
            f"\nNEW: {file_path.name}"
        )

        new_count += 1


    # --------------------------------------------------------
    # Process file
    # --------------------------------------------------------

    records = process_file(
        file_path
    )


    if not records:

        continue


    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    documents = [
        record["document"]
        for record in records
    ]


    print(
        f"  Generating embeddings for "
        f"{len(documents)} chunks..."
    )


    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True
    )


    embeddings = [
        embedding.tolist()
        for embedding in embeddings
    ]


    # --------------------------------------------------------
    # Store in Chroma
    # --------------------------------------------------------

    collection.add(
        ids=[
            record["id"]
            for record in records
        ],

        documents=documents,

        embeddings=embeddings,

        metadatas=[
            record["metadata"]
            for record in records
        ]
    )


    print(
        f"  Stored {len(records)} chunks."
    )


# ============================================================
# DELETE FILES THAT NO LONGER EXIST
# ============================================================

print("\nChecking for deleted files...")


for source in existing_records:

    if source not in current_files:

        print(
            f"\nDELETED FILE: {source}"
        )


        delete_file_chunks(
            source
        )


        deleted_count += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("        INGESTION COMPLETE")
print("========================================")

print(
    f"New files       : {new_count}"
)

print(
    f"Modified files  : {modified_count}"
)

print(
    f"Unchanged files : {unchanged_count}"
)

print(
    f"Deleted files   : {deleted_count}"
)

print(
    f"Total Chroma chunks: "
    f"{collection.count()}"
)

print("========================================")