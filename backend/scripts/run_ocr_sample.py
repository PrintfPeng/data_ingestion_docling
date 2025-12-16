# backend/scripts/run_ocr_sample.py
import sys
import os

# เพิ่ม path เพื่อให้ Python หา module 'ingestion' เจอ
sys.path.append(os.getcwd())

print("🚀 0. Script is loading...")  # <--- ถ้าบรรทัดนี้ไม่ขึ้น แสดงว่ารันผิดไฟล์

try:
    from ingestion.ocr_extractor import ocr_extract_document
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def main():
    print("🚀 1. Start main function")
    
    # --- แก้ Path ตรงนี้ (ใช้ r นำหน้า) ---
    pdf_path = r"C:\Users\ASUS\Downloads\test1.pdf"
    # ------------------------------------
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found at: {pdf_path}")
        return

    print(f"📂 2. Reading file: {pdf_path}")
    print("⏳   Please wait, sending to OCR API (may take 10-30 seconds)...")
    
    try:
        # เรียก OCR
        result = ocr_extract_document(pdf_path)
        
        # ตรวจสอบผลลัพธ์
        if not result.texts:
            print("⚠️ 3. OCR Finished but NO text found.")
        else:
            print(f"✅ 3. OCR Finished. Found content on {len(result.texts)} pages.")
        
        # แสดงข้อความที่อ่านได้
        for page_info in result.texts:
            print("\n" + "=" * 50)
            print(f"📄 Page {page_info['page']}")
            print("=" * 50)
            # โชว์ 500 ตัวอักษรแรก
            print(page_info['content'][:500]) 
            print("\n... (truncated) ...")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

# --- บรรทัดสำคัญที่สุด ห้ามลืม! ---
if __name__ == "__main__":
    main()