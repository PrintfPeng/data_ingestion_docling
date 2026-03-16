# ingestion/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# โหลดค่าจาก .env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# การตั้งค่าสำหรับ Local Model (Qwen2.5 72B)
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY", "no-key")
CUSTOM_API_BASE = os.getenv("CUSTOM_API_BASE", "http://localhost:8000/v1") 
CUSTOM_MODEL_NAME = os.getenv("CUSTOM_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")

# การตั้งค่า OCR API เดิม (จำเป็นต้องมีเพื่อให้สแกนภาพได้)
OCR_API_URL = os.getenv("OCR_API_URL", "https://111.223.37.41:9001")
OCR_USERNAME = os.getenv("OCR_USERNAME", "aiuser")
OCR_PASSWORD = os.getenv("OCR_PASSWORD", "aiuser@S0ftnix")
VERIFY_SSL = os.getenv("VERIFY_SSL", "False").lower() in ("true", "1", "t")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")