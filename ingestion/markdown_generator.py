from __future__ import annotations

"""
markdown_generator.py (Fixed Attribute Name: ingested_at)

รวมความสามารถ:
1. เรียงลำดับตามตำแหน่งจริง (Reading Order)
2. ทำความสะอาดภาษาไทย (Thai Cleaning)
3. จัดรูปแบบตาม AI Role (Header, Warning, Legal, Q&A)
4. ลิงก์รูปภาพคลิกได้ (Clickable Relative Links)
5. สรุปตารางจาก AI (Table Summary)
"""

import re
from pathlib import Path
from typing import List, Any
from dataclasses import dataclass

from .schema import IngestedDocument, TextBlock, TableBlock, ImageBlock

@dataclass
class RenderItem:
    page: int
    y0: float
    x0: float
    type: str  # 'text', 'table', 'image'
    content: Any

def _clean_text_for_markdown(text: str) -> str:
    """
    [Feature 2] ทำความสะอาดภาษาไทย
    """
    if not text: return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    # ลบ space ระหว่างตัวอักษรไทย
    text = re.sub(r'(?<=[\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])', '', text)
    return text.strip()

def _format_text_block(block: TextBlock) -> str:
    """
    [Feature 3] จัดรูปแบบ Markdown ตาม AI Role
    """
    text = _clean_text_for_markdown(block.content)
    if not text: return ""

    role = block.extra.get("role", "normal_text")
    
    if role == "title":
        return f"\n# {text}\n"
    elif role == "section_header":
        return f"\n## {text}\n"
    elif role == "legal_clause":
        return f"\n> ⚖️ **{text}**\n"
    elif role == "warning_note":
        return f"\n> ⚠️ **คำเตือน:** {text}\n"
    elif role == "instruction_step":
        return f"1. {text}"
    elif role == "list_item":
        return f"- {text}"
    elif role == "page_meta":
        return f"\n*(Page {text})*\n"
    elif role == "qna_question":
        return f"\n**Q: {text}**"
    elif role == "qna_answer":
        return f"\n**A: {text}**\n"
    
    return text

def _format_table_block(block: TableBlock) -> str:
    """
    [Feature 4] ตารางพร้อมคำสรุปจาก AI
    """
    md_lines = []
    
    if block.name:
        md_lines.append(f"\n### 📊 ตาราง: {block.name}")
    
    summary = block.extra.get("summary")
    if summary:
        md_lines.append(f"> *AI Summary: {summary}*")
    
    if block.markdown:
        md_lines.append(block.markdown)
    else:
        md_lines.append("*(ตารางไม่มีข้อมูล)*")
    
    md_lines.append("\n")
    return "\n".join(md_lines)

def _format_image_block(block: ImageBlock) -> str:
    """
    [Feature 5] รูปภาพคลิกได้ (Clickable) + Caption AI
    """
    md_lines = []
    
    # ใช้ตัวแปร Local ชื่อ image_name เพื่อไม่ให้สับสน
    image_name = Path(block.file_path).name
    caption = block.caption or "Image"
    
    # Relative Path
    rel_path = f"images/{image_name}"
    
    # Clickable Image
    md_lines.append(f"\n[![{caption}]({rel_path})]({rel_path})")
    
    # Caption
    if block.caption:
        md_lines.append(f"\n*รูปที่: {block.caption}*")
    
    md_lines.append(f"> 📂 File: [`{image_name}`]({rel_path})\n")
    
    return "\n".join(md_lines)

def generate_markdown(doc: IngestedDocument, output_dir: Path):
    """
    [Feature 1] Main Loop: เรียงลำดับตามตำแหน่งจริง
    """
    items: List[RenderItem] = []
    
    for t in doc.texts:
        y0 = t.bbox[1] if t.bbox else 0.0
        x0 = t.bbox[0] if t.bbox else 0.0
        items.append(RenderItem(page=t.page, y0=y0, x0=x0, type='text', content=t))
        
    for tb in doc.tables:
        y0 = tb.bbox[1] if tb.bbox else 0.0
        x0 = tb.bbox[0] if tb.bbox else 0.0
        items.append(RenderItem(page=tb.page, y0=y0, x0=x0, type='table', content=tb))
        
    for im in doc.images:
        y0 = im.bbox[1] if im.bbox else 0.0
        x0 = im.bbox[0] if im.bbox else 0.0
        items.append(RenderItem(page=im.page, y0=y0, x0=x0, type='image', content=im))
        
    # Sort
    items.sort(key=lambda x: (x.page, x.y0, x.x0))
    
    # Build Content
    md_content = []
    
    # [FIXED] ใช้ doc.metadata.file_name และ doc.metadata.ingested_at (ให้ตรงกับ Schema)
    file_name = getattr(doc.metadata, "file_name", "Unknown")
    ingested_at = getattr(doc.metadata, "ingested_at", "Unknown") # แก้ตรงนี้ครับ

    md_content.append(f"# 📄 เอกสาร: {file_name}") 
    md_content.append(f"- **ประเภท:** {doc.metadata.doc_type}")
    md_content.append(f"- **วันที่ประมวลผล:** {ingested_at}")
    md_content.append("---\n")
    
    current_page = 0
    
    for item in items:
        if item.page > current_page:
            current_page = item.page
            md_content.append(f"\n\n--- \n## 📑 Page {current_page}\n\n")
            
        if item.type == 'text':
            md_content.append(_format_text_block(item.content))
        elif item.type == 'table':
            md_content.append(_format_table_block(item.content))
        elif item.type == 'image':
            md_content.append(_format_image_block(item.content))
            
    # Save File
    output_path = output_dir / f"{doc.metadata.doc_id}.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    print(f"✅ [Markdown] Generated Smart Report at: {output_path}")