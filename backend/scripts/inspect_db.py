import sys
import os
import chromadb
import json
from termcolor import colored

# เพิ่ม path ให้ import backend ได้
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Config Path ของ Database
DB_PATH = "./chroma_db"
# ตั้งชื่อ Default ไว้ก่อน (แต่เดี๋ยวเราจะเขียน Logic ให้มันเปลี่ยนเองถ้าไม่เจอ)
TARGET_COLLECTION = "my_collection" 

def inspect_chroma():
    print(colored(f"\n🔍 Inspecting ChromaDB at: {DB_PATH}", "cyan", attrs=["bold"]))
    
    if not os.path.exists(DB_PATH):
        print(colored("❌ Database path not found!", "red"))
        return

    # 1. Connect DB
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
    except Exception as e:
        print(colored(f"❌ Failed to connect to ChromaDB: {e}", "red"))
        return
    
    # List Collections
    collections = client.list_collections()
    collection_names = [c.name for c in collections]
    print(f"📂 Found Collections: {collection_names}")
    
    if not collection_names:
        print(colored("❌ No collections found in database.", "red"))
        return

    # [SMART FIX] เลือก Collection ที่มีอยู่จริงอัตโนมัติ
    if TARGET_COLLECTION in collection_names:
        active_collection_name = TARGET_COLLECTION
    else:
        # ถ้าไม่เจอ my_collection ให้เอาอันแรกที่เจอเลย (เช่น 'documents')
        active_collection_name = collection_names[0]
        print(colored(f"⚠️ Collection '{TARGET_COLLECTION}' not found. Switching to '{active_collection_name}'", "yellow"))

    try:
        collection = client.get_collection(active_collection_name)
    except Exception as e:
        print(colored(f"❌ Error loading collection: {e}", "red"))
        return

    count = collection.count()
    print(colored(f"📊 Total Documents in '{active_collection_name}': {count}", "green"))
    
    if count == 0:
        print("⚠️ Database is empty.")
        return

    # ---------------------------------------------------------
    # 2. ดูตัวอย่างข้อมูลทั่วไป (General Peek)
    # ---------------------------------------------------------
    print(colored("\n[1] 👀 Sample Records (Random 3 items):", "yellow"))
    peek = collection.peek(limit=3)
    
    ids = peek['ids']
    metadatas = peek['metadatas']
    documents = peek['documents']
    
    for i, (doc_id, meta, doc) in enumerate(zip(ids, metadatas, documents)):
        print(f"\n--- Item #{i+1} ---")
        print(colored(f"ID: {doc_id}", "blue"))
        # เช็ค key ที่อาจไม่มีเพื่อกัน error
        src = meta.get('source', 'unknown')
        page = meta.get('page', '?')
        print(f"Type: {src} | Page: {page}")
        print(f"Metadata: {json.dumps(meta, ensure_ascii=False)}")
        print(f"Content (Brief): {doc[:100]}..." if len(doc) > 100 else f"Content: {doc}")

    # ---------------------------------------------------------
    # 3. เจาะจงดูข้อมูลประเภท "ตาราง" (Table Check)
    # ---------------------------------------------------------
    print(colored("\n[2] 📊 Checking Table Data (source='table'):", "yellow"))
    table_results = collection.get(
        where={"source": "table"},
        limit=2
    )
    
    if len(table_results['ids']) > 0:
        for i, (doc_id, meta, doc) in enumerate(zip(table_results['ids'], table_results['metadatas'], table_results['documents'])):
            print(f"\n--- Table #{i+1} ---")
            print(colored(f"ID: {doc_id}", "magenta"))
            print(f"Table Name: {meta.get('category', 'Generic')}")
            print(f"Content (Markdown/Text representation):")
            print(colored(doc[:300] + "...", "white")) 
    else:
        print("❌ No tables found in database.")

    # ---------------------------------------------------------
    # 4. เจาะจงดูข้อมูล Intent (Smart Metadata Check)
    # ---------------------------------------------------------
    print(colored("\n[3] 🧠 Checking Smart Metadata (Intent):", "yellow"))
    # ลองหา intent ทั่วไปที่มีโอกาสเจอเยอะสุดก่อน
    intent_results = collection.get(
        where={"source": "text"}, 
        limit=1
    )
    
    if len(intent_results['ids']) > 0:
        meta = intent_results['metadatas'][0]
        if 'intent' in meta:
            print(f"✅ Found Intent Metadata: {meta['intent']}")
        else:
            # บางทีอาจจะไม่มี intent แต่มีอย่างอื่น
            print(f"⚠️ Intent metadata key missing. Found keys: {list(meta.keys())}")
    else:
        print("⚠️ No text chunks found.")

if __name__ == "__main__":
    inspect_chroma()