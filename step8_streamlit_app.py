"""
Interactive Chat Interface for Document Q&A
Run with: streamlit run step8_streamlit_app.py
"""
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'qa_system_loaded' not in st.session_state:
    st.session_state.qa_system_loaded = False

@st.cache_resource
def load_qa_system():
    """Load the Q&A system (cached so it only loads once)"""
    try:
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load vector database
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Load LLM
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("❌ No GROQ_API_KEY found in .env file!")
            return None, None
        
        llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0.1,
            api_key=api_key
        )
        
        # Create prompt
        template = """You are a helpful assistant answering questions based on the given context.
Use only the information from the context below. If you cannot answer based on the context, say "I cannot find this information in the document."

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create chain
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
        st.error(f"❌ Error loading Q&A system: {e}")
        return None, None

def get_answer(question, chain, retriever):
    """Get answer and sources for a question"""
    try:
        # Get answer
        answer = chain.invoke(question)
        
        # Get source documents
        source_docs = retriever.invoke(question)
        
        return answer, source_docs
    
    except Exception as e:
        return f"Error: {str(e)}", []

# ============================================
# MAIN APP
# ============================================

# Header
st.title("📚 Document Q&A Assistant")
st.markdown("Ask questions about your document and get AI-powered answers with sources!")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses:
    - **Vector Search** to find relevant content
    - **Groq LLM** to generate answers
    - **Source Citations** for transparency
    
    **Features:**
    - ✅ Text extraction
    - ✅ Image/OCR processing
    - ✅ Semantic search
    - ✅ Natural language Q&A
    """)
    
    st.divider()
    
    # Load system button
    if st.button("🔄 Load Q&A System", type="primary"):
        with st.spinner("Loading AI system..."):
            chain, retriever = load_qa_system()
            if chain and retriever:
                st.session_state.chain = chain
                st.session_state.retriever = retriever
                st.session_state.qa_system_loaded = True
                st.success("✅ System loaded!")
            else:
                st.error("❌ Failed to load system")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Sample questions
    st.subheader("💡 Sample Questions")
    sample_questions = [
        "What is this document about?",
        "What are the key findings?",
        "Are there any statistics mentioned?",
        "Summarize the main points",
        "What recommendations are provided?"
    ]
    
    for q in sample_questions:
        if st.button(q, key=f"sample_{q}"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# Main chat interface
if not st.session_state.qa_system_loaded:
    st.info("👈 Click **Load Q&A System** in the sidebar to start!")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(f'<div class="source-box">{source[:400]}...</div>', 
                                  unsafe_allow_html=True)
    
    # Chat input
    if question := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = get_answer(
                    question,
                    st.session_state.chain,
                    st.session_state.retriever
                )
            
            st.markdown(answer)
            
            # Display sources
            if sources:
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(f'<div class="source-box">{doc.page_content[:400]}...</div>', 
                                  unsafe_allow_html=True)
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": [doc.page_content for doc in sources]
        })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 14px;'>
    Built with LangChain, Groq, and Streamlit | 
    Document Q&A System
</div>
""", unsafe_allow_html=True)