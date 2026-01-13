import os
from google import genai
from dotenv import load_dotenv

# โหลด API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: ไม่พบ GOOGLE_API_KEY ใน .env")
    exit()

try:
    client = genai.Client(api_key=api_key)
    print("Checking available models...")
    
    # ดึงรายชื่อโมเดลทั้งหมด
    # หมายเหตุ: คำสั่งนี้อาจจะเปลี่ยนไปตาม version library แต่ลองใช้ standard list ดู
    # สำหรับ google-genai library ใหม่:
    for model in client.models.list():
        # กรองเฉพาะโมเดลที่ generateContent ได้
        if "generateContent" in model.supported_generation_methods:
            print(f"✅ Found: {model.name}") # หรือ model.display_name

except Exception as e:
    print(f"❌ Error: {e}")
    # ถ้า Library ใหม่ใช้ยาก ลองกลับไปใช้ตัวเก่าเช็ค
    print("\n--- Trying Legacy Library ---")
    try:
        import google.generativeai as old_genai
        old_genai.configure(api_key=api_key)
        for m in old_genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"🔹 Legacy Found: {m.name}")
    except ImportError:
        print("Legacy library not installed.")