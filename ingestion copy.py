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

COLLECTION_NAME = "my_documents"
CHROMA_PATH = "./chroma_db"

OCR_ENABLED = True
OCR_DPI = 200

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
}


# ============================================================
# LOAD MODELS
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

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# PATH NORMALIZATION
# ============================================================

def normalize_path(path):

    return str(
        Path(path).resolve()
    ).replace(
        "\\",
        "/"
    ).lower()


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

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# TXT
# ============================================================

def read_txt(file_path):

    try:

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception as e:

        print(
            f"TXT read error: {file_path}"
        )

        print(e)

        return ""


# ============================================================
# PDF + OCR
# ============================================================

def read_pdf(file_path):

    print(
        f"Reading PDF: {file_path.name}"
    )

    text = ""


    # --------------------------------------------------------
    # Normal PDF extraction
    # --------------------------------------------------------

    try:

        reader = PdfReader(
            file_path
        )

        for page in reader.pages:

            try:

                page_text = page.extract_text()

                if page_text:

                    text += (
                        page_text +
                        "\n"
                    )

            except Exception as e:

                print(
                    f"PDF page extraction error: {e}"
                )

    except Exception as e:

        print(
            f"PDF read error: {e}"
        )


    # --------------------------------------------------------
    # Normal text found
    # --------------------------------------------------------

    if text.strip():

        print(
            "Normal PDF text extraction successful."
        )

        return text


    # --------------------------------------------------------
    # OCR fallback
    # --------------------------------------------------------

    if not OCR_ENABLED:

        return ""


    print(
        "No text found. Running OCR..."
    )


    try:

        images = convert_from_path(
            file_path,
            dpi=OCR_DPI
        )

    except Exception as e:

        print(
            f"PDF-to-image error: {e}"
        )

        return ""


    ocr_text = ""


    for page_number, image in enumerate(
        images,
        start=1
    ):

        print(
            f"OCR page "
            f"{page_number}/{len(images)}"
        )

        try:

            page_text = (
                pytesseract.image_to_string(
                    image
                )
            )

            if page_text.strip():

                ocr_text += (
                    page_text +
                    "\n"
                )

        except Exception as e:

            print(
                f"OCR error on page "
                f"{page_number}: {e}"
            )


    if ocr_text.strip():

        print(
            "OCR extraction successful."
        )

    else:

        print(
            "OCR produced no text."
        )


    return ocr_text


# ============================================================
# DOCX
# ============================================================

def read_docx(file_path):

    print(
        f"Reading DOCX: {file_path.name}"
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
# READ FILE
# ============================================================

def read_file(file_path):

    extension = (
        file_path.suffix.lower()
    )


    if extension == ".txt":

        text = read_txt(
            file_path
        )

        return create_text_chunks(
            text
        )


    if extension == ".pdf":

        text = read_pdf(
            file_path
        )

        return create_text_chunks(
            text
        )


    if extension == ".docx":

        try:

            return read_docx(
                file_path
            )

        except Exception as e:

            print(
                f"DOCX error: {e}"
            )

            return []


    return []


# ============================================================
# CREATE RECORDS
# ============================================================

def create_records(
    file_path,
    chunks,
    file_hash
):

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
            "file_type": file_path.suffix.lower(),
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
# DELETE FILE FROM CHROMA
# ============================================================

def delete_file(
    file_path
):

    source = str(
        Path(file_path).resolve()
    )


    print(
        f"Removing from Chroma: {source}"
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
                f"Removed {len(ids)} chunks."
            )

        else:

            print(
                "No Chroma records found."
            )


    except Exception as e:

        print(
            f"Delete error: {e}"
        )


# ============================================================
# INGEST ONE FILE
# ============================================================

def ingest_file(
    file_path
):

    file_path = Path(
        file_path
    )


    if not file_path.exists():

        print(
            f"File does not exist: "
            f"{file_path}"
        )

        return False


    if (
        file_path.suffix.lower()
        not in SUPPORTED_EXTENSIONS
    ):

        print(
            f"Unsupported file: "
            f"{file_path}"
        )

        return False


    print(
        "\n========================================"
    )

    print(
        f"INGESTING: {file_path.name}"
    )

    print(
        "========================================"
    )


    # --------------------------------------------------------
    # Calculate hash
    # --------------------------------------------------------

    try:

        file_hash = (
            calculate_file_hash(
                file_path
            )
        )

    except Exception as e:

        print(
            f"Hash error: {e}"
        )

        return False


    # --------------------------------------------------------
    # Read
    # --------------------------------------------------------

    chunks = read_file(
        file_path
    )


    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk and chunk.strip()
    ]


    if not chunks:

        print(
            "No text found. File was not indexed."
        )

        return False


    print(
        f"Created {len(chunks)} chunks."
    )


    # --------------------------------------------------------
    # Create records
    # --------------------------------------------------------

    records = create_records(
        file_path,
        chunks,
        file_hash
    )


    # --------------------------------------------------------
    # Remove previous version
    # --------------------------------------------------------

    delete_file(
        file_path
    )


    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    documents = [
        record["document"]
        for record in records
    ]


    print(
        "Generating embeddings..."
    )


    try:

        embeddings = (
            embedding_model.encode(
                documents,
                show_progress_bar=True
            )
        )

    except Exception as e:

        print(
            f"Embedding error: {e}"
        )

        return False


    embeddings = [
        embedding.tolist()
        for embedding in embeddings
    ]


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

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
            f"Chroma error: {e}"
        )

        return False


    print(
        f"Successfully stored "
        f"{len(records)} chunks."
    )


    return True


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nIngestion module loaded successfully."
    )

    print(
        "This module is ready for the folder watcher."
    )