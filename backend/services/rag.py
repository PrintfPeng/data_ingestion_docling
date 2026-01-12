import google.generativeai as genai
import json
from pathlib import Path
from ingestion.config import GOOGLE_API_KEY
from .vector_store import search_similar

def answer_question(question: str) -> dict:
    """
    ค้นหาคำตอบและดึงรูปภาพโดยการอ่าน mapping จากไฟล์ image.json
    """
    # 1. Search Vector DB
    try:
        relevant_docs = search_similar(question, k=5)
    except Exception as e:
        print(f"[RAG] Search error: {e}")
        return {"answer": f"Error: {e}", "sources": [], "intent": "error"}
    
    # 2. Prepare Context
    context_text = ""
    sources_data = []
    
    # เก็บ doc_id และ page ที่เกี่ยวข้องไว้ค้นรูป
    # format: { "doc_id": {page1, page2, ...} }
    doc_pages_map = {} 
    
    for doc in relevant_docs:
        content = doc.page_content.replace("\n", " ")
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        
        source_type = meta.get("source", "unknown")
        page = meta.get("page", "?")
        doc_id = meta.get("doc_id") 
        
        # เก็บข้อมูล Source
        sources_data.append({
            "content": content[:300] + "...", 
            "source": source_type,
            "page": page,
            "doc_id": doc_id,
            "metadata": meta 
        })
        context_text += f"--- Source: {source_type.upper()} (Page {page}) ---\n{content}\n\n"

        # บันทึก page ที่เจอเพื่อไปหารูป
        if doc_id and str(page).isdigit():
            if doc_id not in doc_pages_map:
                doc_pages_map[doc_id] = set()
            doc_pages_map[doc_id].add(int(page))

    # 3. [NEW LOGIC] Find Images via image.json
    related_images = []
    processed_urls = set()

    for doc_id, pages in doc_pages_map.items():
        # Path ไปยังไฟล์ image.json ของเอกสารนั้น
        image_json_path = Path("ingested") / doc_id / "image.json"
        
        if image_json_path.exists():
            try:
                # โหลดข้อมูลรูปทั้งหมดของเอกสาร
                images_metadata = json.loads(image_json_path.read_text(encoding="utf-8"))
                
                # คัดกรองเฉพาะรูปที่อยู่หน้าที่เราสนใจ
                for img_item in images_metadata:
                    img_page = img_item.get("page")
                    # เช็คว่ารูปนี้อยู่ในหน้าที่ AI ใช้ตอบคำถามหรือไม่
                    if img_page in pages:
                        # ดึง Path รูปภาพ
                        # ใน json อาจเก็บเป็น "images/doc/img.png" หรือ full path
                        raw_path = img_item.get("file_path") or img_item.get("image_path")
                        if raw_path:
                            # แปลง Path ให้เป็น URL ที่ Frontend เข้าถึงได้
                            # เราต้องตัดให้เหลือแค่ part หลัง 'ingested' หรือใช้ชื่อไฟล์
                            p = Path(raw_path)
                            # สร้าง URL: /ingested/{doc_id}/images/{filename}
                            img_url = f"/ingested/{doc_id}/images/{p.name}"
                            
                            if img_url not in processed_urls:
                                processed_urls.add(img_url)
                                related_images.append({
                                    "url": img_url,
                                    "page": img_page,
                                    "doc_id": doc_id,
                                    "caption": img_item.get("caption", "")
                                })
            except Exception as e:
                print(f"[RAG] Error reading image.json for {doc_id}: {e}")

    # 4. Generate Answer
    if not GOOGLE_API_KEY:
        return {"answer": "Missing API Key", "sources": sources_data, "intent": "config_error"}
        
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        
        prompt = (
            f"Context information is below.\n---------------------\n{context_text}\n---------------------\n"
            f"Answer the query based on the context. If referring to a diagram, describe it from the text context.\n"
            f"Query: {question}\nAnswer (in Thai):"
        )
        
        # Fallback Models logic
        model_candidates = ["gemini-2.5-flash", "gemini-2.0-flash-lite-preview-02-05", "gemini-2.0-flash"]
        answer = ""
        for m in model_candidates:
            try:
                model = genai.GenerativeModel(m)
                answer = model.generate_content(prompt).text
                break
            except: continue
            
        if not answer: answer = "ขออภัย ไม่สามารถประมวลผลคำตอบได้ในขณะนี้"

    except Exception as e:
        answer = f"เกิดข้อผิดพลาด: {str(e)}"

    return {
        "answer": answer,
        "sources": sources_data,
        "related_images": related_images, 
        "intent": "qa"
    }