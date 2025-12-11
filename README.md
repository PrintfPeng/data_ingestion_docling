📄 AI Data Ingestion Pipeline

Automatic PDF → Text → Clean → Semantic Enrich Pipeline (Gemini Powered)

ระบบนี้ถูกออกแบบมาเพื่อช่วยองค์กรในการ แปลงข้อมูลจาก PDF ให้เป็นข้อมูลเชิงโครงสร้าง (structured data) พร้อมการทำ OCR, Cleaning, Table Extraction และ Semantic Enrichment โดยใช้ Google Gemini 2.5 Flash ทำให้สามารถนำข้อมูลไปใช้งานต่อได้ง่าย เช่นส่งเข้า Database, ทำ Data Analysis หรือสร้าง Knowledge Base

🚀 Features
✔ 1) PDF Ingestion

Extract text จาก PDF

Extract ตาราง (table) ด้วย pypdf / pdfplumber

Extract images สำหรับ OCR ภายหลัง

Document validation

✔ 2) OCR (with Google Gemini)

รองรับ OCR ผ่านโมเดล Gemini:

gemini-2.0-flash
gemini-2.5-flash


ระบบจะแปลงแต่ละหน้าเป็นภาพแล้วส่งเข้า Gemini เพื่อดึงข้อความแบบฉลาด

✔ 3) Data Cleaning

Normalize text

Normalize table

Remove noise

Convert to machine-readable format

✔ 4) Semantic Enrichment (AI)

ใช้ Gemini เพื่อ:

Tag sections

Extract semantic meaning

Map relationship ภายในเอกสาร

สร้าง payload พร้อมใช้งานในระบบ downstream (AI agent, LLM, database)

📌 Project Structure
ai-data-ingestion-pipeline/
│
├── ingestion/
│   ├── ocr_extractor.py        # OCR with Gemini
│   ├── document_classifier.py  # Classify PDF type
│   ├── table_extractor.py      # Extract tables
│   └── config.py               # GOOGLE_API_KEY and settings
│
├── cleaning/
│   └── ...
│
├── semantic_enrich/
│   └── ...
│
├── scripts/
│   ├── run_ingestion.py
│   ├── run_cleaning.py
│   └── run_all.py              # Full pipeline
│
├── ingested/                   # Output (ignored by Git)
├── samples/                    # Input samples (ignored by Git)
└── README.md

🔧 Installation
1) Clone project
git clone https://github.com/USERNAME/ai-data-ingestion-pipeline.git
cd ai-data-ingestion-pipeline

2) Install dependencies
pip install -r requirements.txt

3) ตั้งค่า environment variable

สร้างไฟล์ .env:

GOOGLE_API_KEY=YOUR_KEY_HERE

🏃 Running the Pipeline
Run everythingในคำสั่งเดียว
python -m scripts.run_all samples/statement/sample.pdf --doc-id sample --use-gemini

Run เฉพาะ ingestion
python -m scripts.run_ingestion samples/statement/sample.pdf --doc-id sample

Run เฉพาะ cleaning
python -m scripts.run_cleaning --doc-id sample

📂 Output Example

ผลลัพธ์จะถูกเก็บใน:

ingested/sample/
│
├── metadata.json
├── text.json
├── table.json
├── image.json
├── text_clean.json
├── table_clean.json
├── text_enriched.json
└── mapping.json


รองรับ downstream workflows เช่น:

RAG / Knowledge Base

LLM Agent

Analytics Dashboard

Accounting system integration

Internal Data Warehouse

🧠 Technology
Component	Description
Google Gemini 2.0 / 2.5 Flash	OCR + Semantic Enrich
PyMuPDF (fitz)	PDF parsing
pypdf	Table extraction
Python 3.12	Runtime
JSON schema	Structured output
🙌 Author

Peng / PrintfPeng
AI Developer @ Softnix
Building Data, AI, and Multi-Agent systems