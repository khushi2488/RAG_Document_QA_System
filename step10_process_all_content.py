"""
Complete document processor: Text + Tables + Images/OCR
Creates enhanced vector database with ALL content types
"""
from pypdf import PdfReader
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import json

# ============================================
# CONFIGURATION
# ============================================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
PDF_PATH = "qatar_test_doc.pdf"
OUTPUT_FOLDER = "processed_content"

# ============================================
# EXTRACTION FUNCTIONS
# ============================================

def extract_text_with_pages(pdf_path):
    """Extract text with page numbers"""
    print("📄 Extracting text from PDF...")
    reader = PdfReader(pdf_path)
    
    text_content = []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text.strip():
            text_content.append({
                'content': text,
                'page': page_num,
                'type': 'text'
            })
    
    print(f"✅ Extracted text from {len(text_content)} pages")
    return text_content


def extract_tables_with_pages(pdf_path):
    """Extract tables with page numbers"""
    print("📊 Extracting tables from PDF...")
    
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_tables = page.extract_tables()
            
            if page_tables:
                for table_num, table in enumerate(page_tables, 1):
                    if table and len(table) > 0:
                        # Convert to markdown
                        header = table[0]
                        rows = table[1:]
                        
                        markdown = f"Table {table_num} (Page {page_num}):\n\n"
                        markdown += " | ".join(str(h) for h in header) + "\n"
                        markdown += " | ".join(["---"] * len(header)) + "\n"
                        
                        for row in rows:
                            markdown += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                        
                        tables.append({
                            'content': markdown,
                            'page': page_num,
                            'type': 'table',
                            'table_number': table_num
                        })
    
    print(f"✅ Extracted {len(tables)} tables")
    return tables


def extract_images_with_ocr(pdf_path, output_folder):
    """Extract images and run OCR"""
    print("🖼️  Extracting images and running OCR...")
    
    img_folder = os.path.join(output_folder, "images")
    os.makedirs(img_folder, exist_ok=True)
    
    images = []
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Save image
                image_path = f"{img_folder}/page{page_num + 1}_img{img_index + 1}.{image_ext}"
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                # Run OCR
                ocr_text = pytesseract.image_to_string(Image.open(image_path))
                
                if ocr_text.strip():
                    images.append({
                        'content': f"Image OCR (Page {page_num + 1}):\n{ocr_text}",
                        'page': page_num + 1,
                        'type': 'image_ocr',
                        'image_path': image_path
                    })
            except Exception as e:
                print(f"⚠️  Error processing image on page {page_num + 1}: {e}")
    
    pdf_document.close()
    print(f"✅ Processed {len(images)} images with OCR")
    return images


def create_documents_with_metadata(all_content):
    """Convert content to LangChain documents with metadata"""
    print("📝 Creating documents with metadata...")
    
    documents = []
    for item in all_content:
        doc = Document(
            page_content=item['content'],
            metadata={
                'page': item['page'],
                'type': item['type'],
                'source': 'document.pdf'
            }
        )
        documents.append(doc)
    
    print(f"✅ Created {len(documents)} documents")
    return documents


def create_enhanced_vectorstore(documents):
    """Create FAISS vectorstore with all content"""
    print("🔮 Creating enhanced vector database...")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(splits)} chunks")
    
    # Create embeddings
    print("🤖 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector store
    print("💾 Creating FAISS database...")
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    return vectorstore, splits


def save_all_content(all_content, output_folder):
    """Save all extracted content"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Save as JSON
    with open(f"{output_folder}/all_content.json", 'w', encoding='utf-8') as f:
        json.dump(all_content, f, indent=2, ensure_ascii=False)
    
    # Save as readable text
    with open(f"{output_folder}/all_content.txt", 'w', encoding='utf-8') as f:
        for item in all_content:
            f.write(f"\n{'='*80}\n")
            f.write(f"PAGE {item['page']} | TYPE: {item['type']}\n")
            f.write(f"{'='*80}\n")
            f.write(item['content'])
            f.write("\n")
    
    print(f"✅ Saved all content to {output_folder}/")


# ============================================
# MAIN PROCESSING
# ============================================

def main():
    print("="*80)
    print("🚀 COMPLETE DOCUMENT PROCESSING")
    print("="*80 + "\n")
    
    # Extract all content types
    text_content = extract_text_with_pages(PDF_PATH)
    table_content = extract_tables_with_pages(PDF_PATH)
    image_content = extract_images_with_ocr(PDF_PATH, OUTPUT_FOLDER)
    
    # Combine all content
    all_content = text_content + table_content + image_content
    
    print(f"\n📊 SUMMARY:")
    print(f"   Text sections: {len(text_content)}")
    print(f"   Tables: {len(table_content)}")
    print(f"   Images with OCR: {len(image_content)}")
    print(f"   Total items: {len(all_content)}")
    
    # Save all content
    save_all_content(all_content, OUTPUT_FOLDER)
    
    # Create documents
    documents = create_documents_with_metadata(all_content)
    
    # Create enhanced vector database
    vectorstore, splits = create_enhanced_vectorstore(documents)
    
    # Save vector database
    print("💿 Saving enhanced vector database...")
    vectorstore.save_local("faiss_index_enhanced")
    
    print("\n" + "="*80)
    print("🎉 PROCESSING COMPLETE!")
    print("="*80)
    print(f"\n📁 Files created:")
    print(f"   - {OUTPUT_FOLDER}/all_content.json")
    print(f"   - {OUTPUT_FOLDER}/all_content.txt")
    print(f"   - {OUTPUT_FOLDER}/images/ (folder)")
    print(f"   - faiss_index_enhanced/ (vector database)")
    print(f"\n✅ Ready to use with Streamlit app!")


if __name__ == "__main__":
    main()