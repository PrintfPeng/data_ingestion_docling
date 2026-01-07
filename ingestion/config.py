import os
from pathlib import Path
from dotenv import load_dotenv

# 1. โหลด .env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# --- Google Gemini ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- OCR API Configuration ---
# แนะนำ: ลบค่า Default IP/User/Pass ออก เพื่อความปลอดภัย
OCR_API_URL = os.getenv("OCR_API_URL")
OCR_USERNAME = os.getenv("OCR_USERNAME")
OCR_PASSWORD = os.getenv("OCR_PASSWORD")

# --- SSL Verification ---
# Default = False (ไม่ตรวจสอบ SSL) เพื่อให้ง่ายต่อการต่อ IP ภายใน
# ถ้าต้องการเปิด Verify ให้ใส่ VERIFY_SSL=True ใน .env
_verify_ssl_env = os.getenv("VERIFY_SSL", "False").lower()
VERIFY_SSL = _verify_ssl_env in ("true", "1", "t")

# --- Validation (Optional) ---
# เช็คว่าค่าสำคัญมาครบไหม ถ้าไม่ครบให้แจ้งเตือน
if not GOOGLE_API_KEY:
    print("⚠️ Warning: GOOGLE_API_KEY is missing in .env")
if not OCR_PASSWORD:
    print("⚠️ Warning: OCR_PASSWORD is missing in .env")