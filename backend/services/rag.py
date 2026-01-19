# backend/services/rag.py

from google import genai
from google.genai import types
import re
import os
import time
import random
from pathlib import Path
import PIL.Image
from ingestion.config import GOOGLE_API_KEY
from .vector_store import search_similar

# Configuration
# [UPDATED] ใช้ชื่อโมเดลที่ชัวร์ที่สุด (Gemini 2.0 Flash) และระบุ version ย่อย
MODEL_CANDIDATES = [
    "gemini-2.0-flash",       # รุ่นเสถียร (แนะนำ)
    "gemini-2.5-flash",       # รุ่นใหม่ (ถ้ามี)
    "gemini-1.5-flash-001",   # รุ่นเก่าระบุรหัส 001 (มักจะไม่ 404)
    "gemini-1.5-pro-001"      # ตัวสำรองสุดท้าย
]

BASE_DIR = Path(r"D:\DATA_INGES")
INGESTED_PATH = BASE_DIR / "ingested"

# Initialize Client
client = None
if GOOGLE_API_KEY:
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"[RAG] ⚠️ Init Error: {e}")

def generate_with_fallback(client, contents, candidates):
    """ฟังก์ชันช่วยเรียก API แบบ Retry + Fallback พร้อม Log ละเอียด"""
    last_error = None
    
    for model_name in candidates:
        # Retry 2 ครั้งต่อ 1 โมเดล
        for attempt in range(2):
            try:
                # print(f"[RAG] Trying model: {model_name} (Attempt {attempt+1})...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
                return response.text
                
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                
                # ถ้าเป็น Error ชั่วคราว (503, 429) ให้รอแล้วลองใหม่
                if "503" in error_str or "429" in error_str:
                    wait_time = 2 + random.uniform(0, 1)
                    print(f"[RAG] ⚠️ Model {model_name} busy. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue 
                
                # ถ้าเป็น 404 (หาโมเดลไม่เจอ) ให้ข้ามไปตัวถัดไปทันทีไม่ต้อง Retry
                if "404" in error_str:
                    print(f"[RAG] ❌ Model {model_name} not found (404). Skipping...")
                    break 
                
                print(f"[RAG] ❌ Model {model_name} error: {error_str}")
                break
                    
    raise Exception(f"All models failed. Last error: {last_error}")

def answer_question(question: str) -> dict:
    print(f"\n[RAG] 🔍 Query: {question}")
    
    # ---------------------------------------------------------
    # 1. Search Vector DB
    # ---------------------------------------------------------
    try:
        relevant_docs = search_similar(question, k=15) 
    except Exception as e:
        print(f"\n[DEBUG] Found {len(relevant_docs)} docs:")
        for i, d in enumerate(relevant_docs):
            print(f"--- Doc {i+1} (Page {d.metadata.get('page')}) ---\n{d.page_content[:100]}...\n")
            
    context_text = ""
    sources_data = []
    doc_pages_map = {} 
    
    for doc in relevant_docs:
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        content = doc.page_content.replace("\n", " ")
        doc_id = meta.get("doc_id")
        try: page = int(meta.get("page", 0))
        except: page = 0
            
        sources_data.append({
            "content": content[:200] + "...", 
            "page": page,
            "doc_id": doc_id,
            "metadata": meta 
        })
        context_text += f"Document: {doc_id} (Page {page})\nContent: {content}\n\n"

        if doc_id:
            if doc_id not in doc_pages_map: doc_pages_map[doc_id] = set()
            doc_pages_map[doc_id].add(page)

    # ---------------------------------------------------------
    # 2. Hybrid Image Retrieval
    # ---------------------------------------------------------
    related_images = []
    processed_urls = set()
    input_images_for_ai = [] 

    # 2.1 Page Match
    if INGESTED_PATH.exists():
        for doc_id, pages in doc_pages_map.items():
            images_dir = INGESTED_PATH / doc_id / "images"
            if not images_dir.exists(): continue

            for img_file in images_dir.iterdir():
                if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']: continue
                match = re.search(r'(?:_p|page_?)(\d+)', img_file.name)
                if match:
                    try:
                        file_page = int(match.group(1))
                        if file_page in pages:
                            img_url = f"/ingested/{doc_id}/images/{img_file.name}"
                            if img_url not in processed_urls:
                                processed_urls.add(img_url)
                                related_images.append({
                                    "url": img_url,
                                    "page": file_page,
                                    "doc_id": doc_id,
                                    "type": "Page Match"
                                })
                                try:
                                    pil_img = PIL.Image.open(img_file)
                                    input_images_for_ai.append(pil_img)
                                except: pass
                    except: continue

    # 2.2 Semantic Match
    try:
        semantic_image_docs = search_similar(question, k=3, sources=["image"])
        for img_doc in semantic_image_docs:
            meta = img_doc.metadata
            doc_id = meta.get("doc_id")
            file_path = meta.get("file_path") 
            
            if file_path and not os.path.isabs(file_path):
                 candidate_path = BASE_DIR / file_path
                 if not candidate_path.exists():
                     candidate_path = INGESTED_PATH / doc_id / "images" / Path(file_path).name
                 if candidate_path.exists():
                     file_path = str(candidate_path)

            if file_path and os.path.exists(file_path):
                img_name = Path(file_path).name
                img_url = f"/ingested/{doc_id}/images/{img_name}"
                if img_url not in processed_urls:
                    processed_urls.add(img_url)
                    related_images.append({
                        "url": img_url,
                        "page": meta.get("page", 0),
                        "doc_id": doc_id,
                        "type": "Semantic Match"
                    })
                    try:
                        pil_img = PIL.Image.open(file_path)
                        input_images_for_ai.append(pil_img)
                    except: pass
    except: pass

    input_images_for_ai = input_images_for_ai[:5]

    # ---------------------------------------------------------
    # 3. Generate Answer
    # ---------------------------------------------------------
    answer = "ไม่สามารถเชื่อมต่อกับ AI ได้"
    
    if client:
        try:
            prompt_text = (
                f"You are an intelligent assistant analyzing training documents.\n"
                f"Use the Context below to answer the Question in Thai comprehensively.\n"
                f"If the exact answer is split across multiple sections, combine them.\n"
                f"If the answer is found in a table, explain the table data clearly.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )

            contents = [prompt_text]
            contents.extend(input_images_for_ai)

            # ใช้ฟังก์ชัน Fallback ที่อัปเกรดแล้ว
            answer = generate_with_fallback(client, contents, MODEL_CANDIDATES)

        except Exception as e:
            error_msg = str(e)
            print(f"[RAG] ❌ AI Error: {error_msg}")
            
            if "503" in error_msg:
                answer = "ระบบ AI กำลังทำงานหนัก กรุณาลองใหม่ในอีกสักครู่ (Server Overloaded)"
            elif "404" in error_msg:
                answer = "ไม่พบโมเดล AI ที่ระบุ กรุณาตรวจสอบการตั้งค่า (Model Not Found)"
            else:
                answer = f"Error: {error_msg}"

    return {
        "answer": answer,
        "sources": sources_data,
        "related_images": related_images, 
        "intent": "qa"
    }