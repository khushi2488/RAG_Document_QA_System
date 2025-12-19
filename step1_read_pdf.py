from pypdf import PdfReader

# Read the PDF
pdf = PdfReader("qatar_test_doc.pdf")

# Print first page text
print("=== FIRST PAGE TEXT ===")
print(pdf.pages[0].extract_text())

# Print total pages
print(f"\nTotal pages: {len(pdf.pages)}")