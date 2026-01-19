import sys
import os
from pathlib import Path

# จำลองการตั้งค่า Path เหมือนในระบบจริง
root_dir = os.getcwd()
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

print(f"📍 Checking environment from: {root_dir}")
print("-" * 50)

# 1. เช็คว่ามีโฟลเดอร์ scripts และไฟล์ run_ingestion.py หรือไม่
scripts_path = Path(root_dir) / "scripts"
run_ingestion_path = scripts_path / "run_ingestion.py"

if not scripts_path.exists():
    print("❌ ERROR: ไม่พบโฟลเดอร์ 'scripts'")
    exit(1)
if not run_ingestion_path.exists():
    print("❌ ERROR: ไม่พบไฟล์ 'scripts/run_ingestion.py'")
    exit(1)

print("✅ Found scripts/run_ingestion.py")

# 2. ลอง Import และดักจับ Error
print("⏳ Attempting to import 'scripts.run_ingestion'...")
try:
    from scripts.run_ingestion import run_ingestion_pipeline
    print("🎉 SUCCESS! Import สำเร็จ ไม่มีปัญหา")
except ImportError as e:
    print("\n❌ IMPORT ERROR DETECTED!")
    print("-" * 20)
    print(e)
    print("-" * 20)
    print("คำแนะนำเบื้องต้น:")
    if "No module named" in str(e):
        missing_module = str(e).split("'")[1]
        print(f"👉 คุณอาจลืมติดตั้ง Library: '{missing_module}'")
        print(f"👉 ลองรันคำสั่ง: pip install {missing_module}")
    else:
        print("👉 อาจเกิดจากปัญหา Circular Import หรือ Code ในไฟล์ run_ingestion.py มีปัญหา")
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")