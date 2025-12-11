import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ ไม่พบ Key")
else:
    genai.configure(api_key=api_key)
    print(f"Checking models for Key: {api_key[:10]}...")
    
    try:
        print("\n--- Available Generate Content Models ---")
        found_any = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
                found_any = True
        
        if not found_any:
            print("⚠️ ไม่พบโมเดลที่รองรับ generateContent เลย")
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")