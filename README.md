# 📚 RAG-Based Document Q&A System

## 🎯 Project Overview
An AI-powered document question-answering system that uses Retrieval-Augmented Generation (RAG) to extract information from PDFs and answer natural language questions.

## ✨ Features
- ✅ PDF text extraction with page tracking
- ✅ Table detection and extraction
- ✅ Image OCR (Optical Character Recognition)
- ✅ Semantic search using FAISS vector database
- ✅ AI-powered answers using Groq LLM
- ✅ Source citations with page numbers
- ✅ Interactive web interface
- ✅ File upload capability

## 🛠️ Tech Stack
- **Backend**: Python, LangChain, PyPDF, PDFPlumber
- **Vector Database**: FAISS
- **Embeddings**: HuggingFace (BAAI/bge-small-en-v1.5)
- **LLM**: Groq (Llama 3.1 70B)
- **OCR**: Tesseract
- **Frontend**: Streamlit
- **Other**: pytesseract, pdf2image

## 📋 Prerequisites
- Python 3.8+
- Tesseract OCR installed
- Groq API key (free)

## 🚀 Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd RAG_Document_QA_System
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
- **Mac**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr`

### 5. Setup environment variables
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## 📖 Usage

### Step 1: Process a Document
```bash
python src/step1_extract_text.py
python src/step2_create_embeddings.py
```

### Step 2: Run the Web App
```bash
streamlit run src/app.py
```

### Step 3: Use the Interface
1. Upload a PDF or use the sample document
2. Click "Process Document"
3. Ask questions in natural language
4. View AI-generated answers with source citations

## 📊 Sample Questions
- "What is this document about?"
- "What are the key findings?"
- "Are there any statistics mentioned?"
- "Summarize the main points"

## 🏗️ System Architecture
```
User Upload PDF → Text/Table/Image Extraction → 
Chunking → Vector Embeddings (FAISS) → 
User Question → Semantic Search → LLM Processing → 
Answer with Citations
```

<img width="1896" height="880" alt="image" src="https://github.com/user-attachments/assets/16cd98ef-e5dc-4d2c-8529-1e1e3d88164d" />



## 🚧 Current Limitations
- Supports PDF files only (max 200MB)
- OCR accuracy depends on image quality
- Rate limited by Groq API (free tier)

## 👨‍💻 Author
Khushi Patel
https://github.com/khushi2488
