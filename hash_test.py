from pathlib import Path
import hashlib


FILE_PATH = Path(
    r"C:\Users\srira\Documents\local-RAG\test-data\aws.txt"
)


def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while chunk := file.read(4096):

            sha256.update(chunk)

    return sha256.hexdigest()


file_hash = calculate_file_hash(FILE_PATH)


print("File:")
print(FILE_PATH.name)

print("\nSHA256:")
print(file_hash)