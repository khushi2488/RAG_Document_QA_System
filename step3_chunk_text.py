"""
Step 3: Hybrid chunking of Docling markdown output.

Fixed version — the original read every markdown file in the loop but the
actual splitting/saving logic lived OUTSIDE the loop, so only the last file
in the folder ever got chunked. Everything below now runs per-file, inside
the loop, and each document gets its own output folder.
"""

from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ----------------------------
# Config
# ----------------------------

docling_folder = Path("outputs/docling")
chunks_root = Path("outputs/chunks")

headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# ----------------------------
# Process every markdown file
# ----------------------------

markdown_files = list(docling_folder.glob("*.md"))
print(f"Found {len(markdown_files)} markdown files")

for markdown_path in markdown_files:

    print(f"\nProcessing: {markdown_path.name}")

    with open(markdown_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # ----------------------------------------------------
    # Step 1 : Split according to Markdown headings
    # ----------------------------------------------------

    header_docs = markdown_splitter.split_text(markdown_text)
    print(f"Sections found : {len(header_docs)}")

    # ----------------------------------------------------
    # Step 2 : Recursive splitting for large sections
    # ----------------------------------------------------

    final_chunks = []

    for doc in header_docs:

        text = doc.page_content
        metadata = doc.metadata

        if len(text) <= CHUNK_SIZE:
            final_chunks.append(doc)
        else:
            small_chunks = recursive_splitter.create_documents(
                [text],
                metadatas=[metadata],
            )
            final_chunks.extend(small_chunks)

    print(f"Final chunks : {len(final_chunks)}")

    # ----------------------------------------------------
    # Preview first few chunks
    # ----------------------------------------------------

    for i, chunk in enumerate(final_chunks[:5]):
        print("=" * 80)
        print(f"Chunk {i + 1}")
        print("\nMetadata:")
        print(chunk.metadata)
        print("\nContent:")
        print(chunk.page_content[:700])
        print()

    # ----------------------------------------------------
    # Save chunks — one subfolder per source document
    # ----------------------------------------------------

    output_dir = chunks_root / markdown_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(final_chunks, start=1):
        chunk_file = output_dir / f"chunk_{i:03d}.md"
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk.page_content)

    print(f"Saved {len(final_chunks)} chunks to {output_dir}")

print("\nDone. All markdown files processed.")