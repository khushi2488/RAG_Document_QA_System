"""
Extract tables from PDF using pdfplumber
"""
import pdfplumber
import pandas as pd
import json

def extract_tables_from_pdf(pdf_path):
    """Extract all tables from PDF"""
    print(f"📄 Opening PDF: {pdf_path}")
    
    all_tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"📊 Total pages: {len(pdf.pages)}")
        
        for page_num, page in enumerate(pdf.pages, 1):
            print(f"\n🔍 Checking page {page_num}...")
            
            # Extract tables from this page
            tables = page.extract_tables()
            
            if tables:
                print(f"   ✅ Found {len(tables)} table(s)")
                
                for table_num, table in enumerate(tables, 1):
                    # Convert to DataFrame for better handling
                    if table and len(table) > 0:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        
                        # Convert to markdown format for better text representation
                        table_markdown = df.to_markdown(index=False)
                        
                        # Also keep as structured data
                        table_dict = df.to_dict('records')
                        
                        all_tables.append({
                            'page': page_num,
                            'table_number': table_num,
                            'markdown': table_markdown,
                            'data': table_dict,
                            'raw': table
                        })
                        
                        print(f"   📋 Table {table_num}: {len(df)} rows × {len(df.columns)} columns")
            else:
                print(f"   ⚠️  No tables found")
    
    print(f"\n✅ Total tables extracted: {len(all_tables)}")
    return all_tables


def save_tables(tables, output_file="extracted_tables.txt"):
    """Save tables in readable format"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EXTRACTED TABLES\n")
        f.write("="*80 + "\n\n")
        
        for table in tables:
            f.write(f"\n{'='*80}\n")
            f.write(f"PAGE {table['page']} - TABLE {table['table_number']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(table['markdown'])
            f.write("\n\n")
    
    print(f"✅ Saved tables to: {output_file}")


def save_tables_json(tables, output_file="extracted_tables.json"):
    """Save tables as JSON for programmatic access"""
    # Remove raw data for cleaner JSON
    clean_tables = []
    for table in tables:
        clean_tables.append({
            'page': table['page'],
            'table_number': table['table_number'],
            'markdown': table['markdown'],
            'data': table['data']
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_tables, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved tables JSON to: {output_file}")


if __name__ == "__main__":
    pdf_path = "qatar_test_doc.pdf"
    
    print("="*80)
    print("📊 TABLE EXTRACTION")
    print("="*80 + "\n")
    
    # Extract tables
    tables = extract_tables_from_pdf(pdf_path)
    
    if tables:
        # Save in multiple formats
        save_tables(tables)
        save_tables_json(tables)
        
        # Preview first table
        print("\n" + "="*80)
        print("📋 PREVIEW: First Table")
        print("="*80)
        print(f"\nPage: {tables[0]['page']}")
        print(f"Table: {tables[0]['table_number']}\n")
        print(tables[0]['markdown'])
    else:
        print("\n⚠️  No tables found in the document")
    
    print("\n" + "="*80)
    print("✅ TABLE EXTRACTION COMPLETE!")
    print("="*80)