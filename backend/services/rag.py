# backend/services/rag.py

from __future__ import annotations
from google import genai
from google.genai import types
import re
import os
import time
import random
from pathlib import Path
from typing import List, Optional
import PIL.Image

# Import config & vector store
from ingestion.config import GOOGLE_API_KEY
from .vector_store import search_similar

# --- Configuration ---
MODEL_CANDIDATES = [
    "gemini-2.0-flash",       
    "gemini-2.5-flash",       
    "gemini-1.5-flash-001",   
    "gemini-1.5-pro-001"      
]

# กำหนด Path (ปรับให้ตรงกับเครื่องคุณถ้าจำเป็น)
BASE_DIR = Path(r"D:\DATA_INGES")
INGESTED_PATH = BASE_DIR / "ingested"

# --- Initialize Client ---
client = None
if GOOGLE_API_KEY:
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"[RAG] ⚠️ Init Error: {e}")

def generate_with_fallback(client, contents, candidates):
    """ฟังก์ชันช่วยเรียก API แบบ Retry + Fallback"""
    last_error = None
    
    for model_name in candidates:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
                return response.text
                
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                
                if "503" in error_str or "429" in error_str:
                    time.sleep(2)
                    continue 
                if "404" in error_str:
                    break 
                break
                    
    raise Exception(f"All models failed. Last error: {last_error}")

def answer_question(question: str, doc_ids: Optional[List[str]] = None) -> dict:
    """
    ฟังก์ชันตอบคำถาม โดยรองรับการกรอง doc_ids
    """
    print(f"\n[RAG] 🔍 Query: {question} | Filter DocIDs: {doc_ids}")
    
    # ---------------------------------------------------------
    # 1. Search Vector DB (Text)
    # ---------------------------------------------------------
    try:
        # ส่ง doc_ids เข้าไปกรองใน search_similar
        relevant_docs = search_similar(question, k=15, doc_ids=doc_ids) 
    except Exception as e:
        print(f"[RAG] Search Error: {e}")
        relevant_docs = []
            
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

    # 2.1 Page Match (หารูปที่อยู่ในหน้าเดียวกันกับ Text ที่เจอ)
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

    # 2.2 Semantic Match (Image Caption Search)
    try:
        # กรอง doc_ids ตอนค้นหารูปด้วย
        semantic_image_docs = search_similar(question, k=3, sources=["image"], doc_ids=doc_ids)
        for img_doc in semantic_image_docs:
            meta = img_doc.metadata
            doc_id = meta.get("doc_id")
            file_path = meta.get("file_path") 
            
            # แปลง path ให้เป็น absolute path ที่ถูกต้อง
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
    except Exception as e:
        print(f"[RAG] Image Search Error: {e}")

    # จำกัดจำนวนรูปที่จะส่งให้ AI
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
                f"If the answer is found in a table, explain the table data clearly.\n"
                f"If the context is empty or unrelated, please state that no information was found in the selected documents.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )

            contents = [prompt_text]
            contents.extend(input_images_for_ai)

            # ใช้ฟังก์ชัน Fallback
            answer = generate_with_fallback(client, contents, MODEL_CANDIDATES)

        except Exception as e:
            error_msg = str(e)
            print(f"[RAG] ❌ AI Error: {error_msg}")
            answer = f"Error: {error_msg}"

    return {
        "answer": answer,
        "sources": sources_data,
        "related_images": related_images, 
        "intent": "qa"
    }