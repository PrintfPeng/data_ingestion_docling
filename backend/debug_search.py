# debug_search.py
from backend.services.vector_store import search_similar

def main():
    query = "สำนักงานการตรวจเงินแผ่นดิน (สตง.) แสดงความเห็นอย่างไรต่อรายงานการเงินของกรมสรรพากร ประจำปี 2567?"

    docs = search_similar(query=query, k=10)  # ลองดึง 10 อันไปเลย
    print(f"Found {len(docs)} docs\n")

    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        print(f"=== RESULT #{i} ===")
        print("doc_id :", meta.get("doc_id"))
        print("page   :", meta.get("page"))
        print("source :", meta.get("source"))
        print("doc_type:", meta.get("doc_type"))
        print("------- content -------")
        print(d.page_content[:800])  # ตัดมาโชว์ 800 ตัวอักษร
        print("\n")

if __name__ == "__main__":
    main()
