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

REBUILD_COLLECTION = False

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
}

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

EXCLUDED_FILES = set()

COLLECTION_NAME = "my_documents"
CHROMA_PATH = "./chroma_db"

OCR_ENABLED = True
OCR_DPI = 200


# ============================================================
# PATH NORMALIZATION
# ============================================================

def normalize_path(path):

    return str(
        Path(path)
        .resolve()
    ).replace(
        "\\",
        "/"
    ).lower()


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# CHROMA
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


if REBUILD_COLLECTION:

    print("\nRebuilding Chroma collection...")

    try:

        client.delete_collection(
            COLLECTION_NAME
        )

        print(
            "Existing collection deleted."
        )

    except Exception:

        print(
            "No existing collection to delete."
        )


collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

print(
    f"Using collection: {COLLECTION_NAME}"
)


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as file:

        while True:

            data = file.read(
                1024 * 1024
            )

            if not data:
                break

            sha256.update(
                data
            )

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

        print(
            f"  WARNING: Could not read TXT: "
            f"{file_path.name}"
        )

        print(
            f"  Reason: {e}"
        )

        return ""


# ============================================================
# PDF READER + OCR
# ============================================================

def read_pdf(file_path):

    print(
        f"  Reading PDF: {file_path.name}"
    )

    text = ""

    # --------------------------------------------------------
    # NORMAL PDF EXTRACTION
    # --------------------------------------------------------

    try:

        reader = PdfReader(
            file_path
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                page_text = page.extract_text()

                if page_text:

                    text += (
                        page_text +
                        "\n"
                    )

            except Exception as e:

                print(
                    f"    Warning: PDF page "
                    f"{page_number} extraction failed: {e}"
                )

    except Exception as e:

        print(
            "  WARNING: Could not read PDF normally."
        )

        print(
            f"  Reason: {e}"
        )


    # --------------------------------------------------------
    # NORMAL TEXT FOUND
    # --------------------------------------------------------

    if text.strip():

        print(
            "  Normal PDF text extraction successful."
        )

        return text


    # --------------------------------------------------------
    # OCR FALLBACK
    # --------------------------------------------------------

    if not OCR_ENABLED:

        print(
            "  OCR disabled."
        )

        return ""


    print(
        "  No text extracted. Running OCR..."
    )


    try:

        images = convert_from_path(
            file_path,
            dpi=OCR_DPI
        )

    except Exception as e:

        print(
            "  WARNING: Could not convert PDF "
            "to images."
        )

        print(
            f"  Reason: {e}"
        )

        return ""


    ocr_text = ""


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

                ocr_text += (
                    page_text +
                    "\n"
                )

        except Exception as e:

            print(
                f"    WARNING: OCR failed "
                f"on page {page_number}: {e}"
            )


    if ocr_text.strip():

        print(
            "  OCR extraction successful."
        )

    else:

        print(
            "  WARNING: OCR produced no text."
        )


    return ocr_text


# ============================================================
# DOCX READER
# ============================================================

def read_docx(file_path):

    print(
        f"  Reading DOCX: {file_path.name}"
    )

    document = Document(
        file_path
    )

    sections = []

    current_section = []


    def flush_section():

        if not current_section:

            return


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


        try:

            style_name = (
                paragraph.style.name
            )

        except Exception:

            style_name = ""


        is_heading = (
            style_name.startswith(
                "Heading"
            )
            or text.endswith(":")
        )


        if is_heading:

            flush_section()

            current_section.append(
                text
            )

        else:

            current_section.append(
                text
            )


    flush_section()


    return sections


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_text_chunks(text):

    paragraphs = text.split(
        "\n\n"
    )

    chunks = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if paragraph:

            chunks.append(
                paragraph
            )

    return chunks


# ============================================================
# FIND FILES
# ============================================================

def find_files():

    files = []

    successfully_scanned_folders = []


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


        successfully_scanned_folders.append(
            normalize_path(data_folder)
        )


        for file_path in data_folder.rglob("*"):

            if not file_path.is_file():

                continue


            # ------------------------------------------------
            # Excluded directories
            # ------------------------------------------------

            excluded = False


            for part in file_path.parts:

                if part in EXCLUDED_DIRECTORIES:

                    excluded = True

                    break


            if excluded:

                continue


            # ------------------------------------------------
            # Excluded files
            # ------------------------------------------------

            if file_path.name in EXCLUDED_FILES:

                continue


            # ------------------------------------------------
            # Supported extensions
            # ------------------------------------------------

            if (
                file_path.suffix.lower()
                not in SUPPORTED_EXTENSIONS
            ):

                continue


            files.append(
                file_path
            )


    return (
        files,
        successfully_scanned_folders
    )


# ============================================================
# EXISTING CHROMA RECORDS
# ============================================================

def get_existing_records():

    try:

        results = collection.get(
            include=["metadatas"]
        )

    except Exception as e:

        print(
            f"WARNING: Could not read "
            f"Chroma records: {e}"
        )

        return {}


    existing = {}


    for metadata in results.get(
        "metadatas",
        []
    ):

        if not metadata:

            continue


        source = metadata.get(
            "source"
        )

        file_hash = metadata.get(
            "file_hash"
        )


        if source and file_hash:

            normalized_source = (
                normalize_path(source)
            )


            existing[
                normalized_source
            ] = {
                "source": source,
                "file_hash": file_hash,
            }


    return existing


# ============================================================
# PROCESS FILE
# ============================================================

def process_file(file_path):

    extension = (
        file_path.suffix.lower()
    )


    print(
        f"\nProcessing: {file_path}"
    )


    try:

        file_hash = calculate_file_hash(
            file_path
        )

    except Exception as e:

        print(
            f"  WARNING: Could not calculate "
            f"file hash: {e}"
        )

        return []


    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    if extension == ".txt":

        text = read_txt(
            file_path
        )

        chunks = create_text_chunks(
            text
        )


    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    elif extension == ".pdf":

        try:

            text = read_pdf(
                file_path
            )

            chunks = create_text_chunks(
                text
            )

        except Exception as e:

            print(
                f"  WARNING: Could not process "
                f"PDF: {file_path.name}"
            )

            print(
                f"  Reason: {e}"
            )

            return []


    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    elif extension == ".docx":

        try:

            chunks = read_docx(
                file_path
            )

        except Exception as e:

            print(
                f"  WARNING: Could not process "
                f"DOCX: {file_path.name}"
            )

            print(
                f"  Reason: {e}"
            )

            return []


    else:

        return []


    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk and chunk.strip()
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
# DELETE FILE CHUNKS
# ============================================================

def delete_file_chunks(source):

    print(
        f"  Removing old chunks: {source}"
    )


    try:

        results = collection.get(
            where={
                "source": source
            }
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
            f"  WARNING: Could not remove "
            f"old chunks: {e}"
        )


# ============================================================
# MAIN
# ============================================================

print(
    "\nFinding files..."
)


files, scanned_folders = find_files()


print(
    f"\nFound {len(files)} supported files."
)


existing_records = (
    get_existing_records()
)


current_files = set()


new_count = 0
modified_count = 0
unchanged_count = 0
deleted_count = 0
failed_count = 0


# ============================================================
# PROCESS CURRENT FILES
# ============================================================

for file_path in files:

    normalized_source = (
        normalize_path(file_path)
    )


    current_files.add(
        normalized_source
    )


    # --------------------------------------------------------
    # Current file hash
    # --------------------------------------------------------

    try:

        current_hash = (
            calculate_file_hash(
                file_path
            )
        )

    except Exception as e:

        print(
            f"\nWARNING: Could not hash "
            f"{file_path.name}: {e}"
        )

        failed_count += 1

        continue


    # --------------------------------------------------------
    # UNCHANGED
    # --------------------------------------------------------

    if normalized_source in existing_records:

        old_record = (
            existing_records[
                normalized_source
            ]
        )


        if (
            old_record["file_hash"]
            == current_hash
        ):

            print(
                f"\nUNCHANGED: "
                f"{file_path.name}"
            )

            unchanged_count += 1

            continue


    # --------------------------------------------------------
    # PROCESS NEW/MODIFIED FILE
    # --------------------------------------------------------

    if normalized_source in existing_records:

        print(
            f"\nMODIFIED: "
            f"{file_path.name}"
        )

        is_modified = True

    else:

        print(
            f"\nNEW: "
            f"{file_path.name}"
        )

        is_modified = False


    # --------------------------------------------------------
    # Process BEFORE deleting old data
    # --------------------------------------------------------

    records = process_file(
        file_path
    )


    if not records:

        print(
            "  WARNING: File could not be "
            "processed."
        )

        print(
            "  Existing Chroma data will "
            "be preserved."
        )

        failed_count += 1

        continue


    # --------------------------------------------------------
    # Delete old chunks only after successful processing
    # --------------------------------------------------------

    if is_modified:

        old_source = (
            existing_records[
                normalized_source
            ]["source"]
        )


        delete_file_chunks(
            old_source
        )


        modified_count += 1

    else:

        new_count += 1


    # ========================================================
    # EMBEDDINGS
    # ========================================================

    documents = [
        record["document"]
        for record in records
    ]


    print(
        f"  Generating embeddings for "
        f"{len(documents)} chunks..."
    )


    try:

        embeddings = embedding_model.encode(
            documents,
            show_progress_bar=True
        )

    except Exception as e:

        print(
            f"  ERROR generating embeddings: {e}"
        )

        failed_count += 1

        continue


    embeddings = [
        embedding.tolist()
        for embedding in embeddings
    ]


    # ========================================================
    # CHROMA STORAGE
    # ========================================================

    try:

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

    except Exception as e:

        print(
            f"  ERROR storing in Chroma: {e}"
        )

        failed_count += 1

        continue


    print(
        f"  Stored {len(records)} chunks."
    )


# ============================================================
# SAFE DELETED FILE DETECTION
# ============================================================

print(
    "\nChecking for deleted files..."
)


for normalized_source, record in (
    existing_records.items()
):

    # File still exists in current scan
    if normalized_source in current_files:

        continue


    # --------------------------------------------------------
    # Determine whether this source belongs to a folder
    # that was successfully scanned.
    # --------------------------------------------------------

    source_path = Path(
        record["source"]
    )


    normalized_source_path = (
        normalize_path(source_path)
    )


    belongs_to_scanned_folder = False


    for scanned_folder in scanned_folders:

        if (
            normalized_source_path
            == scanned_folder
            or normalized_source_path.startswith(
                scanned_folder + "/"
            )
        ):

            belongs_to_scanned_folder = True

            break


    # --------------------------------------------------------
    # If the folder wasn't successfully scanned,
    # DO NOT delete the Chroma record.
    # --------------------------------------------------------

    if not belongs_to_scanned_folder:

        continue


    # --------------------------------------------------------
    # File genuinely missing from scanned folder
    # --------------------------------------------------------

    print(
        f"\nDELETED FILE: "
        f"{record['source']}"
    )


    delete_file_chunks(
        record["source"]
    )


    deleted_count += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "        INGESTION COMPLETE"
)

print(
    "========================================"
)

print(
    f"New files        : {new_count}"
)

print(
    f"Modified files   : {modified_count}"
)

print(
    f"Unchanged files  : {unchanged_count}"
)

print(
    f"Deleted files    : {deleted_count}"
)

print(
    f"Failed files     : {failed_count}"
)

print(
    f"Total Chroma chunks: "
    f"{collection.count()}"
)

print(
    "========================================"
)