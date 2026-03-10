# 1. ใช้ Python 3.10 เป็นฐาน (เบาและเสถียรสุดสำหรับ AI)
FROM python:3.10-slim

# 2. ตั้งค่าให้โฟลเดอร์หลักในกล่องชื่อ /app
WORKDIR /app

# 3. [สำคัญมาก] ลงโปรแกรมพื้นฐานของ Linux ที่ OpenCV, EasyOCR และ Camelot ต้องใช้!
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# 4. ก๊อปปี้ไฟล์ requirements.txt เข้าไปก่อน แล้วสั่งติดตั้งไลบรารี
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. ก๊อปปี้โค้ดทั้งหมดของมึง (ยกเว้นที่อยู่ใน .dockerignore) เข้าไปในกล่อง
COPY . .

# 6. บอกว่ากล่องนี้ใช้ Port 8000
EXPOSE 8000

# 7. คำสั่งรัน Backend เมื่อกล่องถูกเปิดขึ้นมา
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]