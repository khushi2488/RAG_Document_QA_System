"""
Extract images from PDF and use OCR to read text from them
"""
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF
import os

# ============================================
# CONFIGURATION - UPDATE THESE PATHS IF NEEDED
# ============================================

# For Windows users: Update this path if Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# For Windows users: Update this path to your poppler bin folder
POPPLER_PATH = r'C:\Users\Stark Solutions\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin'  # Update this!

# ============================================

def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    """
    Extract all images from PDF using PyMuPDF
    """
    print(f"📄 Opening PDF: {pdf_path}")
    
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_document = fitz.open(pdf_path)
    image_list = []
    
    print(f"📊 Total pages: {len(pdf_document)}")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        images = page.get_images()
        
        print(f"🔍 Page {page_num + 1}: Found {len(images)} images")
        
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image
            image_filename = f"{output_folder}/page{page_num + 1}_img{img_index + 1}.{image_ext}"
            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)
            
            image_list.append({
                'page': page_num + 1,
                'filename': image_filename
            })
    
    pdf_document.close()
    print(f"✅ Extracted {len(image_list)} images\n")
    return image_list


def perform_ocr_on_image(image_path):
    """
    Use Tesseract OCR to extract text from an image
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return ""


def convert_pdf_pages_to_images(pdf_path, output_folder="page_images"):
    """
    Convert entire PDF pages to images (useful for scanned PDFs)
    """
    print(f"🖼️  Converting PDF pages to images...")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    try:
        # For Windows, uncomment and update the poppler_path:
        # images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        
        # For Mac/Linux:
        images = convert_from_path(pdf_path)
        
        image_paths = []
        for i, image in enumerate(images):
            image_path = f"{output_folder}/page_{i + 1}.png"
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
            print(f"✅ Saved page {i + 1}")
        
        return image_paths
    
    except Exception as e:
        print(f"❌ Error converting PDF to images: {e}")
        print("💡 Make sure poppler is installed and path is correct")
        return []


def main():
    pdf_path = "qatar_test_doc.pdf"
    
    print("="*60)
    print("🔬 IMAGE & OCR EXTRACTION")
    print("="*60)
    
    # Method 1: Extract embedded images from PDF
    print("\n--- METHOD 1: Extract Embedded Images ---")
    images = extract_images_from_pdf(pdf_path)
    
    if images:
        print("\n🔍 Running OCR on extracted images...")
        ocr_results = []
        
        for img_info in images:
            print(f"\nProcessing: {img_info['filename']}")
            text = perform_ocr_on_image(img_info['filename'])
            
            if text:
                print(f"✅ Found text ({len(text)} characters)")
                print(f"Preview: {text[:200]}...")
                
                ocr_results.append({
                    'page': img_info['page'],
                    'filename': img_info['filename'],
                    'text': text
                })
            else:
                print("⚠️  No text found (might be a graphic/chart without text)")
        
        # Save OCR results
        if ocr_results:
            with open("ocr_results.txt", "w", encoding="utf-8") as f:
                f.write("="*60 + "\n")
                f.write("OCR RESULTS FROM IMAGES\n")
                f.write("="*60 + "\n\n")
                
                for result in ocr_results:
                    f.write(f"\n--- PAGE {result['page']} - {result['filename']} ---\n")
                    f.write(result['text'] + "\n")
            
            print(f"\n✅ Saved OCR results to: ocr_results.txt")
    
    # Method 2: Convert entire pages to images (for scanned PDFs)
    print("\n\n--- METHOD 2: Full Page OCR (for scanned documents) ---")
    print("⚠️  This is slower but works for scanned PDFs")
    
    choice = input("Run full page OCR? (y/n): ").strip().lower()
    
    if choice == 'y':
        page_images = convert_pdf_pages_to_images(pdf_path)
        
        if page_images:
            print("\n🔍 Running OCR on full pages...")
            page_ocr_results = []
            
            for i, img_path in enumerate(page_images, 1):
                print(f"\nProcessing page {i}...")
                text = perform_ocr_on_image(img_path)
                
                if text:
                    print(f"✅ Extracted {len(text)} characters")
                    page_ocr_results.append({
                        'page': i,
                        'text': text
                    })
            
            # Save full page OCR results
            if page_ocr_results:
                with open("full_page_ocr.txt", "w", encoding="utf-8") as f:
                    f.write("="*60 + "\n")
                    f.write("FULL PAGE OCR RESULTS\n")
                    f.write("="*60 + "\n\n")
                    
                    for result in page_ocr_results:
                        f.write(f"\n=== PAGE {result['page']} ===\n")
                        f.write(result['text'] + "\n")
                
                print(f"\n✅ Saved full page OCR to: full_page_ocr.txt")
    
    print("\n" + "="*60)
    print("✅ IMAGE & OCR EXTRACTION COMPLETE!")
    print("="*60)
    print("\n📁 Check these files:")
    print("   - extracted_images/ folder (embedded images)")
    print("   - ocr_results.txt (text from embedded images)")
    if choice == 'y':
        print("   - page_images/ folder (full page images)")
        print("   - full_page_ocr.txt (text from full pages)")


if __name__ == "__main__":
    main()