"""
This creates a FAISS vector database from your PDF text
Run this ONCE before creating the chat interface
"""
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

print("🔄 Starting embedding creation process...")

# Step 1: Load the extracted text
print("📄 Loading extracted text...")
try:
    with open("extracted_text.txt", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create a Document object
    documents = [Document(page_content=content, metadata={"source": "qatar_test_doc.pdf"})]
    print(f"✅ Loaded document")
    print(f"   Total characters: {len(content)}")
except FileNotFoundError:
    print("❌ Error: extracted_text.txt not found!")
    print("   Run step2_extract_all.py first!")
    exit(1)

# Step 2: Split into chunks
print("✂️  Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Each chunk = 1000 characters
    chunk_overlap=200     # 200 character overlap for context
)
texts = text_splitter.split_documents(documents)
print(f"✅ Created {len(texts)} text chunks")

# Step 3: Load FREE embedding model
print("🤖 Loading FREE embedding model...")
print("   ⏳ First time: Will download ~440MB model (takes 2-5 minutes)")
print("   ⏳ After that: Loads instantly from cache")
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",  # Smaller, faster model
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Embedding model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("💡 Run: pip install sentence-transformers")
    exit(1)

# Step 4: Create FAISS vector database
print("💾 Creating FAISS vector database...")
print("   ⏳ Converting all chunks to embeddings... (1-3 minutes)")
try:
    vectorstore = FAISS.from_documents(
        documents=texts,
        embedding=embeddings
    )
    print("✅ FAISS database created!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Run: pip install faiss-cpu")
    exit(1)

# Step 5: Save to disk
print("💿 Saving to disk...")
try:
    vectorstore.save_local("faiss_index")
    print("✅ Saved to 'faiss_index' folder")
except Exception as e:
    print(f"❌ Error saving: {e}")
    exit(1)

print("\n" + "="*50)
print("🎉 EMBEDDINGS CREATED SUCCESSFULLY!")
print("="*50)
print(f"📊 Summary:")
print(f"   - Text chunks: {len(texts)}")
print(f"   - Embedding model: BAAI/bge-small-en-v1.5")
print(f"   - Database location: faiss_index/")
print(f"   - Status: Ready to use!")
print("\n📌 Next: Run step5_ask_questions.py")
print("="*50)