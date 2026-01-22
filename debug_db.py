import chromadb
import os

# กำหนด Path ของ Database โดยตรง (ปกติจะอยู่ที่ folder chroma_db หน้าบ้าน)
DB_PATH = os.path.join(os.getcwd(), "chroma_db")

def inspect_document(target_doc_id):
    print(f"\n🔍 Inspecting Document ID: '{target_doc_id}'")
    print(f"📂 Database Path: {DB_PATH}")
    print("="*60)
    
    if not os.path.exists(DB_PATH):
        print(f"❌ ไม่พบโฟลเดอร์ Database ที่: {DB_PATH}")
        return

    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection("document_chunks")
        
        # ค้นหาเฉพาะ ID นี้
        result = collection.get(
            where={"doc_id": target_doc_id},
            include=["documents", "metadatas"]
        )
        
        ids = result['ids']
        total_found = len(ids)
        print(f"✅ Found total chunks: {total_found}")
        
        if total_found == 0:
            print("❌ ไม่พบข้อมูลใน Database เลย (Ingestion อาจล้มเหลว หรือใช้ชื่อ ID ไม่ตรง)")
            return

        # สุ่มแสดงเนื้อหา 3 ชิ้นแรก
        print("\n--- ตัวอย่างเนื้อหา (First 3 chunks) ---")
        for i in range(min(3, total_found)):
            meta = result['metadatas'][i]
            content = result['documents'][i]
            
            print(f"\n📄 Chunk #{i+1}")
            print(f"   Page: {meta.get('page')}")
            print(f"   Source: {meta.get('source')}")
            print(f"   Content Preview: {content[:300]!r}") 
            
            if not content.strip():
                print("   ⚠️ WARNING: Content is EMPTY!")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"❌ Error accessing database: {e}")

if __name__ == "__main__":
    # ใส่ชื่อ ID ที่ต้องการตรวจสอบ (ต้องตรงกับตอน Upload)
    inspect_document("AI-DURIAN-สมบูรณ์")