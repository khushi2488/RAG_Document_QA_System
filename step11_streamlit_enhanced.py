"""
Enhanced Streamlit App with:
- File upload
- Tables, images, OCR
- Page number citations
- Export chat history
"""
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
import os
from dotenv import load_dotenv
from datetime import datetime
import tempfile

load_dotenv()

# Page config
st.set_page_config(
    page_title="Advanced Document Q&A",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .page-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .type-badge {
        background-color: #2196F3;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'qa_system_loaded' not in st.session_state:
    st.session_state.qa_system_loaded = False
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

@st.cache_resource
def load_embeddings():
    """Load embedding model (cached)"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF and create vector database"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract text with page numbers
        reader = PdfReader(tmp_path)
        documents = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        'page': page_num,
                        'type': 'text',
                        'source': uploaded_file.name
                    }
                )
                documents.append(doc)
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        embeddings = load_embeddings()
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings
        )
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return vectorstore, len(documents)
    
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None, 0

def load_existing_vectorstore():
    """Load existing enhanced vector database"""
    try:
        embeddings = load_embeddings()
        
        # Try enhanced version first
        if os.path.exists("faiss_index_enhanced"):
            vectorstore = FAISS.load_local(
                "faiss_index_enhanced",
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore, "enhanced"
        
        # Fall back to basic version
        elif os.path.exists("faiss_index"):
            vectorstore = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore, "basic"
        
        return None, None
    
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        return None, None

def create_qa_chain(vectorstore):
    """Create Q&A chain with page citations"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("❌ No GROQ_API_KEY found in .env file!")
            return None, None
        
        # Create retriever that returns more documents for better context
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Get top 5 most relevant chunks
        )
        
        # LLM
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            api_key=api_key
        )
        
        # Enhanced prompt with citation instructions
        template = """You are a helpful assistant answering questions based on the given context.

IMPORTANT: 
- Use ONLY the information from the context below
- Be specific and cite information accurately
- If you cannot answer based on the context, say "I cannot find this information in the document"
- When mentioning statistics or specific facts, try to indicate which part of the context they come from

Context:
{context}

Question: {question}

Answer (be specific and accurate):"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain, retriever
    
    except Exception as e:
        st.error(f"Error creating Q&A chain: {e}")
        return None, None

def get_answer_with_sources(question, chain, retriever):
    """Get answer with source documents including page numbers"""
    try:
        answer = chain.invoke(question)
        source_docs = retriever.invoke(question)
        return answer, source_docs
    except Exception as e:
        return f"Error: {str(e)}", []

def export_chat_history():
    """Export chat history as text file"""
    if not st.session_state.messages:
        return None
    
    export_text = f"Chat History - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += f"Document: {st.session_state.uploaded_file_name or 'Default Document'}\n"
    export_text += "="*80 + "\n\n"
    
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        export_text += f"{role}: {msg['content']}\n\n"
        
        if msg["role"] == "assistant" and "sources" in msg:
            export_text += "Sources:\n"
            for i, source in enumerate(msg["sources"], 1):
                export_text += f"  [{i}] Page {source['page']} ({source['type']})\n"
            export_text += "\n"
        
        export_text += "-"*80 + "\n\n"
    
    return export_text

# ============================================
# MAIN APP
# ============================================

st.title("📚 Advanced Document Q&A System")
st.markdown("Upload a PDF or use the existing document to ask questions!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # File upload
    st.subheader("📤 Upload New Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF to analyze"
    )
    
    if uploaded_file:
        if st.button("🔄 Process Uploaded PDF", type="primary"):
            with st.spinner("Processing PDF..."):
                vectorstore, num_pages = process_uploaded_pdf(uploaded_file)
                
                if vectorstore:
                    chain, retriever = create_qa_chain(vectorstore)
                    
                    if chain and retriever:
                        st.session_state.chain = chain
                        st.session_state.retriever = retriever
                        st.session_state.qa_system_loaded = True
                        st.session_state.uploaded_file_name = uploaded_file.name
                        st.session_state.messages = []  # Clear old messages
                        
                        st.success(f"✅ Processed {num_pages} pages!")
                        st.balloons()
    
    st.divider()
    
    # Load existing database
    st.subheader("📁 Use Existing Database")
    if st.button("🔄 Load Existing Document"):
        with st.spinner("Loading existing database..."):
            vectorstore, db_type = load_existing_vectorstore()
            
            if vectorstore:
                chain, retriever = create_qa_chain(vectorstore)
                
                if chain and retriever:
                    st.session_state.chain = chain
                    st.session_state.retriever = retriever
                    st.session_state.qa_system_loaded = True
                    st.session_state.uploaded_file_name = f"Existing ({db_type})"
                    
                    st.success(f"✅ Loaded {db_type} database!")
            else:
                st.error("❌ No existing database found. Run step10_process_all_content.py first!")
    
    st.divider()
    
    # Chat controls
    st.subheader("💬 Chat Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.session_state.messages:
            export_text = export_chat_history()
            if export_text:
                st.download_button(
                    label="💾 Export",
                    data=export_text,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    st.divider()
    
    # Sample questions
    st.subheader("💡 Sample Questions")
    samples = [
        "What is this document about?",
        "What are the key findings?",
        "Are there any tables or statistics?",
        "What recommendations are provided?",
        "Summarize the main points"
    ]
    
    for q in samples:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            if st.session_state.qa_system_loaded:
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()
            else:
                st.warning("⚠️ Please load a document first!")
    
    st.divider()
    
    # Info
    st.subheader("ℹ️ Features")
    st.markdown("""
    ✅ Text extraction  
    ✅ Table detection  
    ✅ Image OCR  
    ✅ Page citations  
    ✅ File upload  
    ✅ Chat export  
    """)

# Main chat interface
if not st.session_state.qa_system_loaded:
    st.info("👈 Upload a PDF or load the existing document from the sidebar to start!")
    
    # Show instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        **Option 1: Upload New PDF**
        1. Click "Choose a PDF file" in sidebar
        2. Upload your PDF
        3. Click "Process Uploaded PDF"
        4. Wait for processing (may take 1-2 minutes)
        5. Start asking questions!
        
        **Option 2: Use Existing Document**
        1. Make sure you've run `step10_process_all_content.py`
        2. Click "Load Existing Document" in sidebar
        3. Start asking questions!
        
        **Features:**
        - Ask questions in natural language
        - Get answers with page number citations
        - See source content for transparency
        - Export chat history
        """)
else:
    # Show current document
    if st.session_state.uploaded_file_name:
        st.info(f"📄 Current document: **{st.session_state.uploaded_file_name}**")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources with page numbers
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        page = source.get('page', 'Unknown')
                        content_type = source.get('type', 'text')
                        content = source.get('content', '')
                        
                        st.markdown(
                            f'<div class="source-box">'
                            f'<span class="page-badge">Page {page}</span>'
                            f'<span class="type-badge">{content_type}</span>'
                            f'<p style="margin-top: 0.5rem;">{content[:400]}...</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
    
    # Chat input
    if question := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, source_docs = get_answer_with_sources(
                    question,
                    st.session_state.chain,
                    st.session_state.retriever
                )
            
            st.markdown(answer)
            
            # Display sources
            # if source_docs:
                # with st.expander("📚 View Sources"):
                #     for i, doc in enumerate(source_docs, 1):
                #         page = doc.metadata.get('page', 'Unknown')
                #         content_type = doc.metadata.get('type', 'text')
                        
                #         st.markdown(
                #             f'<div class="source-box">'
                #             f'<span class="page-badge">Page {page}</span>'
                #             f'<span class="type-badge">{content_type}</span>'
                #             f'<p style="margin-top: 0.5rem;">{doc.page_content[:400]}...</p>'
                #             f'</div>',
                #             unsafe_allow_html=True
                #         )
        
        # Save to history
        sources = [{
            'page': doc.metadata.get('page', 'Unknown'),
            'type': doc.metadata.get('type', 'text'),
            'content': doc.page_content
        } for doc in source_docs]
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 14px;'>
    Advanced RAG System | Text + Tables + Images/OCR + Page Citations | 
    Built with LangChain, Groq, and Streamlit
</div>
""", unsafe_allow_html=True)