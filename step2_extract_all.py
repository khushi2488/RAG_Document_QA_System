from pypdf import PdfReader
import pandas as pd

def extract_text_from_pdf(pdf_path):
    """Extract all text from PDF"""
    reader = PdfReader(pdf_path)
    all_text = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        all_text.append({
            'page': page_num + 1,
            'text': text
        })
    
    return all_text

# Run it
content = extract_text_from_pdf("qatar_test_doc.pdf")

# Save to a text file so you can see what you got
with open("extracted_text.txt", "w", encoding="utf-8") as f:
    for item in content:
        f.write(f"\n=== PAGE {item['page']} ===\n")
        f.write(item['text'])

print("✓ Text extracted! Check extracted_text.txt file")