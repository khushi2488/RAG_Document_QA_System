"""
Step 1: Read PDF data.

Fixed version — the original had three nested loops re-globbing and
re-opening the same PDFs. This does the same job in a single pass.
"""

from pathlib import Path
from pypdf import PdfReader

dataset_folder = Path("Dataset")

pdf_files = sorted(dataset_folder.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF(s):")
for pdf_path in pdf_files:
    print(" ", pdf_path)

for pdf_path in pdf_files:
    print("=" * 80)
    print(f"Processing: {pdf_path.name}")

    pdf = PdfReader(pdf_path)
    print("Pages:", len(pdf.pages))

    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            print(f"Page {i + 1}: {len(text)} characters")
        else:
            print(f"Page {i + 1}: NO TEXT")