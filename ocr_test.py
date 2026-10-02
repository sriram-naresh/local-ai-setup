from pathlib import Path
from pdf2image import convert_from_path
import pytesseract

PDF_FILE = Path(r"C:\Users\srira\Documents\local-RAG\test-data\saanvika-passport.pdf")

print("Converting PDF to image...")

images = convert_from_path(
    PDF_FILE,
    dpi=200,
    first_page=1,
    last_page=1
)

print(f"Pages converted: {len(images)}")

print("Running OCR...")

text = pytesseract.image_to_string(images[0])

print("\n===== OCR RESULT =====\n")
print(text)

print("\n===== END =====")