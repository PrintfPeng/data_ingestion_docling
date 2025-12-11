# test_google_key.py
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def main() -> None:
    # ให้รู้ก่อนว่าตอนนี้เราอยู่โฟลเดอร์ไหน
    print("CWD:", Path().resolve())

    # โหลดค่าใน .env และทับค่าเก่าใน env (กัน Ghost Key)
    load_dotenv(override=True)

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    print("API_KEY_RAW:", repr(api_key))

    if not api_key:
        print("!!! NO GOOGLE_API_KEY / GEMINI_API_KEY in environment")
        return

    # ย้ำอีกที ใส่กลับเข้า env ให้ lib อื่นใช้ key เดียวกัน
    os.environ["GOOGLE_API_KEY"] = api_key

    # สร้าง client สำหรับ embeddings
    emb = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )

    # ลอง embed ข้อความสั้น ๆ ดู
    vec = emb.embed_query("hello world")
    print("Embedding length:", len(vec))
    print("First 5 values:", vec[:5])


if __name__ == "__main__":
    main()
