"""
Streamlit RAG Chatbot
----------------------
Upload PDFs -> parse -> chunk -> embed (BGE-small) -> store in Qdrant (in-memory)
-> retrieve -> answer with Groq LLM.

Run with:
    streamlit run app.py

Requires a GROQ_API_KEY (set in a .env file, environment variable, or entered
directly in the sidebar).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_groq import ChatGroq

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
COLLECTION_NAME = "rag_chunks"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
LLM_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a helpful AI assistant.

Answer the user's question ONLY using the provided context.

Instructions:
- Use only the information present in the context.
- Do not make up facts.
- If the answer is not found in the context, reply:
  "I couldn't find the answer in the provided documents."
- Be clear, concise, and accurate."""

st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")

# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str) -> ChatGroq:
    return ChatGroq(model=LLM_MODEL, temperature=0.1, api_key=api_key)


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #

def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return a list of (page_number, text) for a PDF, skipping unreadable pages."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text))
    return pages


def chunk_document(pdf_name: str, pages: list[tuple[int, str]]) -> list[Document]:
    """Split each page's text into overlapping chunks, tagged with source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs: list[Document] = []
    for page_num, text in pages:
        if not text.strip():
            continue
        for i, chunk in enumerate(splitter.split_text(text)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "document": pdf_name,
                        "page": page_num,
                        "chunk_id": f"{pdf_name}_p{page_num}_c{i + 1}",
                    },
                )
            )
    return docs


def build_vectorstore(all_docs: list[Document]) -> tuple[QdrantVectorStore, QdrantClient]:
    """Embed all chunks and store them in a fresh in-memory Qdrant collection."""
    embeddings = get_embeddings()
    client = QdrantClient(":memory:")
    client.create_collection(
        COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(
        client=client, collection_name=COLLECTION_NAME, embedding=embeddings
    )
    vectorstore.add_documents(all_docs)
    return vectorstore, client


def build_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        parts.append(
            f"Document {i}\n"
            f"Source: {d.metadata.get('document')} (page {d.metadata.get('page')})\n\n"
            f"{d.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, docs: list[Document]) -> str:
    context = build_context(docs)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"======================\nCONTEXT\n======================\n{context}\n\n"
        f"======================\nQUESTION\n======================\n{question}\n\n"
        f"======================\nANSWER\n======================\n"
    )


# --------------------------------------------------------------------------- #
# Sidebar: setup + document ingestion
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("Setup")

    api_key_input = st.text_input(
        "GROQ_API_KEY",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Loaded from .env if present. You can also paste it here.",
    )

    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    process_clicked = st.button("Process documents", disabled=not uploaded_files)

    if process_clicked:
        with st.spinner("Parsing, chunking, and embedding documents..."):
            all_docs: list[Document] = []
            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                try:
                    pages = extract_pages(tmp_path)
                    all_docs.extend(chunk_document(f.name, pages))
                finally:
                    os.unlink(tmp_path)

            if not all_docs:
                st.error("No extractable text found in the uploaded PDF(s).")
            else:
                vectorstore, client = build_vectorstore(all_docs)
                st.session_state.vectorstore = vectorstore
                st.session_state.qdrant_client = client
                st.session_state.num_chunks = len(all_docs)
                st.session_state.doc_names = sorted(
                    {d.metadata["document"] for d in all_docs}
                )
                st.session_state.messages = []

        if "num_chunks" in st.session_state:
            st.success(
                f"Indexed {st.session_state.num_chunks} chunks from "
                f"{len(st.session_state.doc_names)} document(s)."
            )

    if "doc_names" in st.session_state:
        st.caption("Indexed documents:")
        for name in st.session_state.doc_names:
            st.caption(f"• {name}")

    top_k = st.slider("Chunks to retrieve (k)", 1, 10, DEFAULT_TOP_K)

    if st.button("Reset session"):
        for key in ("vectorstore", "qdrant_client", "num_chunks", "doc_names", "messages"):
            st.session_state.pop(key, None)
        st.rerun()

# --------------------------------------------------------------------------- #
# Main chat area
# --------------------------------------------------------------------------- #

st.title("📄 RAG Chatbot")

if "vectorstore" not in st.session_state:
    st.info("Upload PDFs and click **Process documents** in the sidebar to get started.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['document']}**, page {s['page']}")

question = st.chat_input("Ask a question about your documents...")

if question:
    if not api_key_input:
        st.error("Please enter your GROQ_API_KEY in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": top_k}
            )
            docs = retriever.invoke(question)
            prompt = build_prompt(question, docs)
            llm = get_llm(api_key_input)
            answer = llm.invoke(prompt).content

        st.markdown(answer)
        sources = [
            {"document": d.metadata.get("document"), "page": d.metadata.get("page")}
            for d in docs
        ]
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- **{s['document']}**, page {s['page']}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )