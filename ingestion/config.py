import os
from pathlib import Path
from dotenv import load_dotenv

# โหลด .env ถ้ามี
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# --- Google Gemini (เก็บไว้ก่อนเผื่อใช้ในส่วนอื่น) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- OCR API (Tesseract / EasyOCR Configuration) ---
# URL จากในรูป Swagger (ใช้ HTTPS)
OCR_API_URL = os.getenv("OCR_API_URL", "https://111.223.37.41:9001")

# Username/Password จากในรูป Line
OCR_USERNAME = os.getenv("OCR_USERNAME", "aiuser")
OCR_PASSWORD = os.getenv("OCR_PASSWORD", "aiuser@S0ftnix")

# ปิดการตรวจสอบ SSL เพราะเว็บเป็น Not Secure (Self-signed certificate)
VERIFY_SSL = False