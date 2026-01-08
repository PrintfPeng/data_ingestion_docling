# ก๊อปเฉพาะฟังก์ชันนี้ไปทับใน ingestion/table_extractor.py (ช่วงท้ายไฟล์)
def extract_tables(
    file_path: str | Path,
    doc_id: str,
    doc_type: str = "generic",
    pages: str = "all",
    flavor_priority: Optional[list[str]] = None,
) -> List[TableBlock]:
    path = Path(file_path)
    if not path.exists(): raise FileNotFoundError(f"PDF file not found: {path}")

    gemini_vision = _get_gemini_model("gemini-2.0-flash")
    all_tables: List[TableBlock] = []
    
    # --- STRATEGY 1: Vision ---
    if gemini_vision:
        try:
            doc = fitz.open(path)
            # (ตัดโค้ดส่วนเตรียม page_indices ... ให้เหมือนเดิม)
            page_indices = range(len(doc)) # แบบย่อ
            
            table_counter = 0
            for page_idx in page_indices:
                # ... (Logic การหาตารางเหมือนเดิม) ...
                
                # [FIXED] เพิ่ม Delay ตรงนี้ เป็น 10 วินาที กัน Quota เต็ม
                if table_counter > 0:
                    print("[table_extractor] Sleeping 10s to avoid rate limit...")
                    time.sleep(10)
                
                # ... (ส่วนประมวลผล Vision เหมือนเดิม) ...
        except Exception:
            pass

    # --- STRATEGY 2: Camelot ---
    # ... (ส่วน Camelot เหมือนเดิม ไม่ต้องแก้) ...
    
    return all_tables