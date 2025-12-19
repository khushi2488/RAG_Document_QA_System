"""
FREE Question Answering using Groq API - Modern LangChain syntax
"""
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

print("🔄 Setting up Q&A system...")

# Step 1: Get API key from .env
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ No API key found in .env file!")
    print("Add this line to your .env file:")
    print("GROQ_API_KEY=your_key_here")
    exit(1)

# Step 2: Load embeddings and vector database
print("\n🔄 Loading vector database...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = FAISS.load_local(
    "faiss_index", 
    embeddings,
    allow_dangerous_deserialization=True
)
print("✅ Database loaded!")

# Step 3: Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 4: Connect to Groq LLM
print("🤖 Connecting to Groq LLM...")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    api_key=api_key
)
print("✅ LLM connected!")

# Step 5: Create prompt template
template = """You are a helpful assistant answering questions based on the given context.
Use only the information from the context below. If you cannot answer based on the context, say "I cannot find this information in the document."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Step 6: Create the chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ Q&A system ready!\n")

# Step 7: Interactive Q&A loop
print("="*60)
print("💬 ASK QUESTIONS ABOUT YOUR DOCUMENT")
print("="*60)
print("Type 'quit' to exit\n")

while True:
    question = input("❓ Your question: ").strip()
    
    if question.lower() in ['quit', 'exit', 'q']:
        print("👋 Goodbye!")
        break
    
    if not question:
        continue
    
    print("\n🔍 Searching and generating answer...\n")
    
    try:
        # Get answer
        answer = chain.invoke(question)
        
        # Get source documents
        source_docs = retriever.invoke(question)
        
        # Print answer
        print("💡 ANSWER:")
        print("-" * 60)
        print(answer)
        print("-" * 60)
        
        # Print sources
        print("\n📚 SOURCES:")
        for i, doc in enumerate(source_docs, 1):
            print(f"\n[Source {i}]")
            print(doc.page_content[:300] + "...")
        
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your Groq API key is correct\n")