"""
Step 4:
Create embeddings from chunk files and store them in Qdrant.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

print("=" * 60)
print("Creating Embeddings using BGE-Small")
print("=" * 60)

# -------------------------------------------------------
# Step 1 : Read all chunk files
# -------------------------------------------------------

chunks_root = Path("outputs/chunks")

documents = []

print("Loading chunk files...")

for document_folder in chunks_root.iterdir():

    if not document_folder.is_dir():
        continue

    for chunk_file in sorted(document_folder.glob("chunk_*.md")):

        text = chunk_file.read_text(encoding="utf-8")

        metadata = {
            "document": document_folder.name,
            "chunk": chunk_file.stem,
            "source": str(chunk_file),
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

print(f"Loaded {len(documents)} chunks.")

# -------------------------------------------------------
# Step 2 : Load Embedding Model
# -------------------------------------------------------

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Embedding model loaded.")

# -------------------------------------------------------
# Step 3 : Create Local Qdrant Database
# -------------------------------------------------------

print("\nInitializing Qdrant...")

client = QdrantClient(path="qdrant_db")

collection_name = "rag_chunks"

# recreate collection each run

try:
    client.delete_collection(collection_name)
except Exception:
    pass

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,          # BGE-small embedding dimension
        distance=Distance.COSINE,
    ),
)

print("Qdrant collection created.")

# -------------------------------------------------------
# Step 4 : Store embeddings
# -------------------------------------------------------

print("\nGenerating embeddings and storing vectors...")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

vectorstore.add_documents(documents)

print("Embeddings stored successfully.")
client.close()
# -------------------------------------------------------
# Summary
# -------------------------------------------------------

print("\n" + "=" * 60)
print("Embedding Creation Completed")
print("=" * 60)

print(f"Embedding Model : BAAI/bge-small-en-v1.5")
print(f"Vector Database : Qdrant")
print(f"Chunks Indexed  : {len(documents)}")
print(f"Collection      : {collection_name}")
print(f"Database Folder : qdrant_db/")
print("=" * 60)