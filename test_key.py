import os
from dotenv import load_dotenv
import google.generativeai as genai

print("--- Diagnostic Start ---")
load_dotenv(override=True)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: ไม่พบ Key")
else:
    masked_key = f"{api_key[:8]}...{api_key[-5:]}"
    print(f"🔑 Key ที่ใช้: {masked_key}")
    
    print("running Test API call...")
    try:
        genai.configure(api_key=api_key)
        # แก้ตรงนี้: ใช้ gemini-1.5-flash แทน gemini-pro
        model = genai.GenerativeModel('gemini-1.5-flash') 
        response = model.generate_content("Hello")
        print(f"✅ API Test Success! Gemini replied: {response.text}")
    except Exception as e:
        print(f"❌ API Test Failed: {e}")

print("--- Diagnostic End ---")