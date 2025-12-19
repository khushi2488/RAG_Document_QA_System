from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
# from langchain.chains import RetrievalQA

print("🔄 Loading vector database...")

# Load embeddings model (same as before)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load saved FAISS database
vectorstore = FAISS.load_local(
    "faiss_index", 
    embeddings,
    allow_dangerous_deserialization=True
)
print("✅ Database loaded!")

# Test: Search for relevant chunks
print("\n🔍 Testing search...")
question = "What is this document about?"
docs = vectorstore.similarity_search(question, k=3)

print(f"\nQuestion: {question}")
print(f"Found {len(docs)} relevant chunks:")
for i, doc in enumerate(docs, 1):
    print(f"\n--- Chunk {i} ---")
    print(doc.page_content[:200] + "...")