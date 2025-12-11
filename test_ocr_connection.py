from ingestion.ocr_extractor import ocr_page_via_api, _get_api_token
import fitz

def test_connection():
    print("--- Testing OCR API Connection ---")
    
    # 1. Test Login
    try:
        token = _get_api_token()
        print(f"✅ Login Success! Token: {token[:10]}...")
    except Exception as e:
        print(f"❌ Login Failed: {e}")
        return

    # 2. Test OCR (ถ้ามีไฟล์ sample.pdf)
    try:
        pdf_path = "samples/statement/sample.pdf" # ตรวจสอบ path ไฟล์ให้ถูก
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        
        print(f"Sending Page 1 image ({len(img_bytes)} bytes) to OCR API...")
        text = ocr_page_via_api(img_bytes)
        
        print("\n--- OCR Result ---")
        print(text[:500]) # โชว์ 500 ตัวอักษรแรก
        print("------------------")
        print("✅ OCR Process Finished")
        
    except Exception as e:
        print(f"❌ OCR Process Failed: {e}")

if __name__ == "__main__":
    test_connection()