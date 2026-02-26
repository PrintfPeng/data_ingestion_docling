# backend/debug_search.py
import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from backend.services.vector_store import search_similar

def test_search(query: str, k: int = 20):
    print(f"\n🔍 กำลังค้นหา: '{query}'")
    try:
        # ดึงมา 20 Chunk แรกมาดูว่าตัวที่ต้องการติดมาด้วยไหม
        results = search_similar(query, k=k)
        print(f"✅ พบข้อมูล {len(results)} Chunks (เรียงตามคะแนนความเหมือนดิบ):")
        print("-" * 70)
        
        for i, doc in enumerate(results, 1):
            md = doc.metadata or {}
            source = md.get("source", "unknown")
            page = md.get("page", "?")
            doc_id = md.get("doc_id", "unknown")
            
            # ตัดเนื้อหามาพรีวิวแค่ 150 ตัวอักษร ลบขึ้นบรรทัดใหม่เพื่อให้อ่านง่าย
            content = doc.page_content.replace("\n", " ").strip()
            preview = content[:150] + ("..." if len(content) > 150 else "")
            
            print(f"[{i:02d}] Source: {source:10} | Page: {page:<4} | Doc: {doc_id[:20]}...")
            print(f"     Preview: {preview}\n")
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการค้นหา: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # คำถามเจ้าปัญหา
        query = "กองทุนเงินทดแทนในปี 2558 มีจำนวนลูกจ้างกี่คน"
        
    test_search(query, k=20)