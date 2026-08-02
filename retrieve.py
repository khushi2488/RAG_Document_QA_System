"""
Step 5:
Retrieve relevant chunks from Qdrant using semantic search.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

print("=" * 60)
print("Semantic Retrieval from Qdrant")
print("=" * 60)

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5


# -------------------------------------------------------
# Step 1 : Load Embedding Model
# -------------------------------------------------------

def load_embedding_model():
    """Load the embedding model."""

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Embedding model loaded.")

    return embeddings


# -------------------------------------------------------
# Step 2 : Connect to Qdrant
# -------------------------------------------------------

def connect_qdrant():
    """Connect to the local Qdrant database."""

    print("\nConnecting to Qdrant...")

    client = QdrantClient(path=QDRANT_PATH)

    print("Connected to Qdrant.")

    return client


# -------------------------------------------------------
# Step 3 : Create Vector Store
# -------------------------------------------------------

def create_vectorstore(client, embeddings):
    """Create LangChain Vector Store."""

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    return vectorstore


# -------------------------------------------------------
# Step 4 : Create Retriever
# -------------------------------------------------------

def create_retriever(vectorstore):
    """Create retriever."""

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    return retriever


# -------------------------------------------------------
# Step 5 : Retrieve Documents
# -------------------------------------------------------

def retrieve_documents(retriever, query):
    """Retrieve relevant documents."""

    print("\nSearching...\n")

    documents = retriever.invoke(query)

    return documents


# -------------------------------------------------------
# Step 6 : Display Results
# -------------------------------------------------------

def display_results(documents):
    """Print retrieved chunks."""

    print("=" * 60)
    print(f"Retrieved {len(documents)} Chunks")
    print("=" * 60)

    if not documents:
        print("No relevant chunks found.")
        return

    for i, doc in enumerate(documents, start=1):

        print(f"\nChunk {i}")
        print("-" * 60)

        print(f"Document : {doc.metadata.get('document')}")
        print(f"Chunk    : {doc.metadata.get('chunk')}")
        print(f"Source   : {doc.metadata.get('source')}")

        print("\nContent:\n")
        print(doc.page_content)

        print("-" * 60)


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    embeddings = load_embedding_model()

    client = connect_qdrant()

    vectorstore = create_vectorstore(client, embeddings)

    retriever = create_retriever(vectorstore)

    print("\nType 'exit' to quit.\n")

    while True:

        query = input("Ask a question: ").strip()

        if query.lower() == "exit":
            break

        documents = retrieve_documents(retriever, query)

        display_results(documents)

    client.close()

    print("\nQdrant connection closed.")


# -------------------------------------------------------

if __name__ == "__main__":
    main()