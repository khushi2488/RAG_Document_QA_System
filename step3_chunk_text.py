# from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter



# Read the extracted text
with open("extracted_text.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# Create a splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Each piece = 1000 characters
    chunk_overlap=200  # Overlap = 200 characters
)

# Split the text
chunks = splitter.split_text(full_text)

# Print to see
print(f"Total chunks created: {len(chunks)}")
print("\n=== FIRST CHUNK ===")
print(chunks[0])
print("\n=== SECOND CHUNK ===")
print(chunks[1])