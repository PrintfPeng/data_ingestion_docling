import streamlit as st
import sys
import os

# Fix Import Path ให้มองเห็น backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.services.vector_store import VectorStore
from backend.services.image_embedder import ImageEmbedder

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Durian RAG", layout="wide")

st.title("📚 AI Intelligent Document Assistant")
st.caption("ระบบค้นหาเอกสารอัจฉริยะ (Text + Image Retrieval)")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    n_text = st.slider("จำนวนข้อความที่ค้นหา", 1, 10, 3)
    n_image = st.slider("จำนวนรูปภาพที่ค้นหา", 1, 5, 2)
    st.divider()
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.success("Cleared!")

# --- Load Models (Cache ไว้จะได้ไม่โหลดใหม่ทุกครั้งที่กดปุ่ม) ---
@st.cache_resource
def load_resources():
    print("Loading VectorStore & Embedder...")
    vs = VectorStore()
    ie = ImageEmbedder()
    return vs, ie

try:
    vector_store, image_embedder = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# --- Main UI ---
query = st.text_input("💬 พิมพ์คำถามของคุณเกี่ยวกับเอกสาร:", placeholder="เช่น ขอขั้นตอนการวางแผนฝึกอบรม หรือ ตารางการฝึกอบรม")

if query:
    with st.spinner('กำลังค้นหาข้อมูล...'):
        # 1. ค้นหา Text
        text_results = vector_store.query_similar_documents(query, n_results=n_text)
        
        # 2. ค้นหา Image
        query_vec = image_embedder.embed_text(query)
        image_results = vector_store.query_similar_images(query_vec, n_results=n_image)

    # --- แสดงผลลัพธ์ ---
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("📄 เนื้อหาที่เกี่ยวข้อง (Text)")
        if text_results['documents'][0]:
            for i, doc in enumerate(text_results['documents'][0]):
                meta = text_results['metadatas'][0][i]
                with st.expander(f"📌 แหล่งที่มา: {meta.get('file_name', 'Doc')} (Score: {text_results['distances'][0][i]:.4f})", expanded=True):
                    st.markdown(doc)
        else:
            st.info("ไม่พบข้อมูลเนื้อหา")

    with col2:
        st.subheader("🖼️ รูปภาพประกอบ (Images)")
        if image_results['ids'][0]:
            for i, img_path in enumerate(image_results['ids'][0]): # ID คือชื่อไฟล์
                meta = image_results['metadatas'][0][i]
                real_path = meta.get('image_path')
                score = image_results['distances'][0][i]
                
                if real_path and os.path.exists(real_path):
                    st.image(real_path, caption=f"Score: {score:.4f}", use_container_width=True)
                else:
                    st.warning(f"File not found: {real_path}")
        else:
            st.info("ไม่พบรูปภาพที่เกี่ยวข้อง")