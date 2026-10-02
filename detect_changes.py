from pathlib import Path
import hashlib


DATA_FOLDER = Path(
    r"C:\Users\srira\Documents\local-RAG\test-data"
)


# Simulated hashes from a previous scan
# Later, these will come from ChromaDB.
previous_hashes = {
    "aws.txt": "old_hash",
    "kubernetes.txt": "some_hash",
}


def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while chunk := file.read(4096):
            sha256.update(chunk)

    return sha256.hexdigest()


for file_path in DATA_FOLDER.rglob("*.txt"):

    current_hash = calculate_file_hash(file_path)

    file_name = file_path.name

    print(f"\nFile: {file_name}")

    # New file
    if file_name not in previous_hashes:

        print("Status: NEW")

    # Existing file
    else:

        old_hash = previous_hashes[file_name]

        if current_hash == old_hash:

            print("Status: UNCHANGED")

        else:

            print("Status: MODIFIED")