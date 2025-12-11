import os
from pathlib import Path
from dotenv import load_dotenv

# โหลด .env ถ้ามี
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
