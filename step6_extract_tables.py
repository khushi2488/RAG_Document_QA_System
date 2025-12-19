"""
Extract tables from PDF using pdfplumber
"""
import pdfplumber
import json

print("🔄 Starting table extraction...\n")

# Open the PDF
pdf_path = "qatar_test_doc.pdf"

try:
    with pdfplumber.open(pdf_path) as pdf:
        all_tables = []
        
        print(f"📄 Processing {len(pdf.pages)} pages...\n")
        
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract tables from this page
            tables = page.extract_tables()
            
            if tables:
                print(f"✅ Page {page_num}: Found {len(tables)} table(s)")
                
                for table_num, table in enumerate(tables, start=1):
                    # Store table info
                    table_data = {
                        'page': page_num,
                        'table_number': table_num,
                        'rows': len(table),
                        'columns': len(table[0]) if table else 0,
                        'data': table
                    }
                    all_tables.append(table_data)
                    
                    # Print preview
                    print(f"   Table {table_num}: {len(table)} rows × {len(table[0]) if table else 0} columns")
            else:
                print(f"⚪ Page {page_num}: No tables found")
        
        print(f"\n📊 Total tables extracted: {len(all_tables)}")
        
        # Save tables to JSON file
        if all_tables:
            with open("extracted_tables.json", "w", encoding="utf-8") as f:
                json.dump(all_tables, f, indent=2, ensure_ascii=False)
            print("💾 Saved to: extracted_tables.json")
            
            # Also save human-readable version
            with open("extracted_tables.txt", "w", encoding="utf-8") as f:
                for table_info in all_tables:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"PAGE {table_info['page']} - TABLE {table_info['table_number']}\n")
                    f.write(f"{'='*60}\n")
                    
                    table = table_info['data']
                    for row in table:
                        f.write(" | ".join(str(cell) if cell else "" for cell in row))
                        f.write("\n")
            
            print("📄 Saved readable version to: extracted_tables.txt")
            
            # Show first table as preview
            if all_tables:
                print("\n" + "="*60)
                print("🔍 PREVIEW: First Table")
                print("="*60)
                first_table = all_tables[0]['data']
                for i, row in enumerate(first_table[:5]):  # Show first 5 rows
                    print(" | ".join(str(cell)[:20] if cell else "" for cell in row))
                if len(first_table) > 5:
                    print(f"... ({len(first_table) - 5} more rows)")
        else:
            print("\n⚠️  No tables found in this PDF")
            print("💡 This is normal if your PDF has no tables")
        
except FileNotFoundError:
    print("❌ Error: document.pdf not found!")
    print("💡 Make sure document.pdf is in the same folder")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n✅ Table extraction complete!")