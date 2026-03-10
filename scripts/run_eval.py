import asyncio
import pandas as pd
import os
import sys
import time
import re
import json
import logging
from pathlib import Path

# ปิด Error รกๆ ของ LiteLLM
import litellm
from litellm import completion
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

# --- SETUP PATH & ENV ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from backend.services.rag import answer_question

# --- CONFIGURATION (Hybrid System) ---
PRIMARY_MODEL = "openai/qwen/qwen-2.5-72b-instruct"
BACKUP_MODEL = "gemini/gemini-2.5-flash"

# Setup Primary (Custom Qwen)
if os.getenv("CUSTOM_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("CUSTOM_API_KEY")
    os.environ["OPENAI_API_BASE"] = os.getenv("CUSTOM_API_BASE", "http://111.223.37.51/v1")
    print(f"🔵 Primary Judge: {PRIMARY_MODEL}")

# Setup Backup (Google Gemini)
if os.getenv("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    print(f"🟠 Backup Judge: {BACKUP_MODEL}")

# =====================================================================
# 📝 ชุดคำถามทดสอบ 30 ข้อ (สำหรับเอกสารใบสมัครงานสหกิจศึกษา)
# =====================================================================
DOC_ID = "015_co-op_03_ใบสมัครงานสหกิจศึกษา_update" 

TEST_CASES = [
    # --- 🟢 ระดับ EASY (10 ข้อ) ---
    {"level": "easy", "question": "นักศึกษาที่สมัครงานในเอกสารนี้ชื่ออะไร?", "expected": "นายซอลาฮูดิน ดอเล๊าะ"},
    {"level": "easy", "question": "รหัสนักศึกษาของผู้สมัครคืออะไร?", "expected": "406559015"},
    {"level": "easy", "question": "นักศึกษาเรียนอยู่ชั้นปีที่เท่าไหร่ และได้เกรดเฉลี่ยรวมเท่าใด?", "expected": "นักศึกษาชั้นปีที่ 4 เกรดเฉลี่ยรวม 3.22"},
    {"level": "easy", "question": "นักศึกษาต้องการสมัครงานกับสถานประกอบการชื่ออะไร?", "expected": "Better Soft Co., Ltd."},
    {"level": "easy", "question": "สมัครงานในตำแหน่งอะไร?", "expected": "front end developer"},
    {"level": "easy", "question": "สถานประกอบการตั้งอยู่ที่จังหวัดอะไร?", "expected": "กรุงเทพมหานคร"},
    {"level": "easy", "question": "สาขาวิชาและคณะที่นักศึกษาเรียนอยู่คืออะไร?", "expected": "สาขาวิชาวิทยาการคอมพิวเตอร์และเทคโนโลยีดิจิทัล คณะวิทยาศาสตร์เทคโนโลยีและการเกษตร"},
    {"level": "easy", "question": "บัตรประจำตัวประชาชนของผู้สมัครหมดอายุวันที่เท่าไหร่?", "expected": "8 เมษายน 2576"},
    {"level": "easy", "question": "นักศึกษานับถือศาสนาอะไร?", "expected": "อิสลาม"},
    {"level": "easy", "question": "ระยะเวลาปฏิบัติงานสหกิจศึกษาเริ่มและสิ้นสุดเมื่อไหร่?", "expected": "เริ่มวันที่ 3 พฤศจิกายน 2568 ถึงวันที่ 16 มีนาคม 2569"},

    # --- 🟡 ระดับ MEDIUM (10 ข้อ) ---
    {"level": "medium", "question": "อาชีพเป้าหมายที่นักศึกษาสนใจมีอะไรบ้าง?", "expected": "1. Programmer 2. Cyber Security 3. Software Engineer"},
    {"level": "medium", "question": "ทักษะด้านเทคโนโลยีที่ผู้สมัครควรมีสำหรับตำแหน่งนี้คืออะไรบ้าง?", "expected": "HTML, CSS, JavaScript, Framework (React/Vue.js), Responsive Design, Git"},
    {"level": "medium", "question": "ลักษณะงานนี้ต้องทำงานร่วมกับทีมใดบ้าง?", "expected": "ทีม UX/UI และ Back-End"},
    {"level": "medium", "question": "ผู้สมัครจบระดับมัธยมปลายจากสถานศึกษาใด และได้ผลการศึกษาเท่าไหร่?", "expected": "โรงเรียนอิสลามบูรพาวิทยา ผลการศึกษา 2.80"},
    {"level": "medium", "question": "ผู้สมัครเคยเข้าร่วมการอบรมโครงการ Yala Hackathon เมื่อวันที่เท่าไหร่?", "expected": "13/08/68 - 15/08/68"},
    {"level": "medium", "question": "โครงการ Super AI Engineer 5th year จัดขึ้นที่จังหวัดใด?", "expected": "มอ.ภูเก็ต"},
    {"level": "medium", "question": "การอบรมเรื่อง AI for Office Automation จัดโดยหน่วยงานใด?", "expected": "มหาวิทยาลัยราชภัฏยะลา"},
    {"level": "medium", "question": "สรุปรายละเอียดลักษณะงาน (Job Description) ที่นักศึกษาต้องทำมาให้หน่อย", "expected": "ออกแบบพัฒนาเว็บโดยใช้ HTML/CSS/JS และ Framework, ทำงานกับทีม UX/UI และ Back-End, เชื่อมต่อ API เป็นต้น"},
    {"level": "medium", "question": "สรุปประวัติการอบรมทั้งหมดของนักศึกษาคนนี้ว่าเคยผ่านหัวข้ออะไรมาบ้าง", "expected": "AI for Office Automation, Super AI Engineer 5th year, และ Yala Hackathon"},
    {"level": "medium", "question": "เอกสารฉบับนี้คือเอกสารอะไร และมีจุดประสงค์เพื่ออะไร?", "expected": "ใบสมัครงานสหกิจศึกษา เพื่อใช้สมัครงานในตำแหน่ง front end developer"},

    # --- 🔴 ระดับ HARD (10 ข้อ) ---
    {"level": "hard", "question": "ขอดูตารางประวัติการศึกษาของผู้สมัครหน่อย", "expected": "AI ตอบกลับโดยใช้ Tag [SHOW_TABLE:TBL_x] ที่แสดงประวัติการศึกษา"},
    {"level": "hard", "question": "ขอดูตารางประวัติการอบรมของผู้สมัครหน่อย", "expected": "AI ตอบกลับโดยใช้ Tag [SHOW_TABLE:TBL_x] ที่แสดงตารางการอบรม"},
    {"level": "hard", "question": "ขอดูตารางความสามารถทางภาษาของผู้สมัคร", "expected": "AI ตอบกลับโดยใช้ Tag [SHOW_TABLE:TBL_x] ที่แสดงความสามารถทางภาษาอังกฤษ มลายู ไทย"},
    {"level": "hard", "question": "ในเอกสารมีตารางที่แสดงผลการเรียนตอนมัธยมไหม ขอดูหน่อย", "expected": "AI ตอบกลับโดยใช้ Tag [SHOW_TABLE:TBL_x] หรือดึงข้อมูลตารางประวัติการศึกษามาแสดง"},
    {"level": "hard", "question": "ช่วยแสดงตารางที่บอกว่าเคยไปอบรมที่ไหนบ้างให้ดูหน่อย", "expected": "AI ตอบกลับโดยใช้ Tag [SHOW_TABLE:TBL_x] แสดงตารางการอบรม"},
    {"level": "hard", "question": "มีตารางประเมินทักษะการฟังพูดอ่านเขียนภาษาต่างๆ ไหม ขอดูตารางนั้น", "expected": "AI ตอบกลับโดยใช้ Tag [SHOW_TABLE:TBL_x] แสดงตารางความสามารถทางภาษา"},
    {"level": "hard", "question": "ขอดูรูปถ่ายของนักศึกษาที่สมัครงานหน่อย", "expected": "AI ตอบกลับโดยใช้ Tag [SHOW_IMAGE: ...] เพื่อแสดงรูปถ่ายนักศึกษา"},
    {"level": "hard", "question": "ในเอกสารมีรูปภาพใบหน้าของผู้สมัครไหม ขอดูภาพนั้นหน่อย", "expected": "AI ตอบกลับโดยใช้ Tag [SHOW_IMAGE: ...] เพื่อแสดงรูปถ่ายบุคคล"},
    {"level": "hard", "question": "จากข้อมูลในเอกสาร ทักษะ Soft Skill หรือลักษณะนิสัยการทำงานที่ผู้สมัครควรมีสำหรับตำแหน่งนี้คืออะไร?", "expected": "มีความรับผิดชอบและพร้อมทำงานเป็นทีมได้อย่างมีประสิทธิภาพ"},
    {"level": "hard", "question": "วิเคราะห์จากสายงานที่สมัครและอาชีพเป้าหมาย คิดว่านักศึกษาคนนี้ถนัดงานด้านไหน?", "expected": "สายงานด้านการพัฒนาซอฟต์แวร์ เขียนโปรแกรม (Programmer, Software Engineer, Front End)"}
]
# =====================================================================
# 🧠 AI Judge Prompt
# =====================================================================
JUDGE_PROMPT = """
คุณคือผู้เชี่ยวชาญด้านการประเมินระบบ AI (RAG Evaluator)
หน้าที่ของคุณคือ ให้คะแนนความถูกต้องของ 'คำตอบจาก AI' โดยเปรียบเทียบกับ 'คำตอบที่คาดหวัง'

⚠️ **กฎเหล็กสำหรับการประเมินระบบ Hybrid (มีรูปภาพและตาราง)** ⚠️
1. ถ้าระบบตอบกลับมาโดยมี Tag รูปภาพ เช่น `[SHOW_IMAGE: ...]` ถือว่าระบบทำงาน **สมบูรณ์แบบ** (ให้ 5 คะแนน)
2. ถ้าระบบตอบกลับมาโดยมี Tag ตาราง เช่น `[SHOW_TABLE: ...]` ถือว่าระบบทำงาน **สมบูรณ์แบบ** (ให้ 5 คะแนน)
3. หากคำถามไม่ได้ขอดูรูปหรือตาราง ให้ประเมินจากความถูกต้องของเนื้อหา (Fact-checking) ตามปกติ

เกณฑ์การให้คะแนน (1-5):
1: ตอบผิดไปคนละเรื่อง, ข้อมูลมั่ว (Hallucination) หรือตอบว่าหาไม่เจอ
3: ตอบถูกบางส่วน แต่ขาดรายละเอียดสำคัญไป
5: ตอบถูกต้อง ครบถ้วน หรือมีการเรียกใช้ [SHOW_IMAGE] / [SHOW_TABLE] ได้ตรงตามคำถาม

---
คำถามที่ผู้ใช้ถาม: {question}
คำตอบที่คาดหวัง: {expected}
คำตอบที่ AI ตอบกลับมา: {answer}
---

กรุณาวิเคราะห์และตอบกลับมาเป็น JSON Format เท่านั้น (ห้ามมีข้อความอื่นปน):
{{
    "score_correctness": <int 1-5>,
    "judge_reason": "<คำอธิบายเหตุผลสั้นๆ ว่าทำไมถึงให้คะแนนเท่านี้>"
}}
"""

def parse_json_response(raw_text: str) -> dict:
    try:
        clean_text = re.sub(r'```(?:json)?', '', raw_text).strip()
        return json.loads(clean_text)
    except:
        return {"score_correctness": 1, "judge_reason": f"Failed to parse AI output: {raw_text[:50]}..."}

async def evaluate_single_turn(test_case: dict, index: int) -> dict:
    question = test_case["question"]
    expected = test_case["expected"]
    level = test_case["level"]

    print(f"\n▶️ [{index}/30] Q: {question}")
    start_time = time.time()
    
    # 1. รับคำตอบจากระบบ RAG ของเรา 
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = await answer_question(query=question, doc_ids=[DOC_ID])
    
    answer = result.get("answer", "No answer generated.")
    latency = time.time() - start_time
    
    print(f"   🤖 A: {answer[:150].replace(chr(10), ' ')}...")

    # 2. เริ่มการให้คะแนน
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, answer=answer)
    judge_res = None
    
    # 🌟 แผน A: พยายามใช้ Custom Qwen เป็นกรรมการก่อน
    if os.getenv("CUSTOM_API_KEY"):
        try:
            import sys, io
            backup_stdout = sys.stdout
            sys.stdout = io.StringIO() 
            
            response = completion(model=PRIMARY_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=200, response_format={"type": "json_object"})
            
            sys.stdout = backup_stdout 
            judge_res = parse_json_response(response.choices[0].message.content)
            
        except Exception:
            sys.stdout = backup_stdout 
            print("   🔄 [Auto-Fallback] Qwen ไม่ตอบสนอง สลับให้ Gemini ตรวจแทน...")

    # 🌟 แผน B: ถ้าแผน A พัง สลับไปใช้ Gemini ให้คะแนนอัตโนมัติ
    if not judge_res and os.getenv("GOOGLE_API_KEY"):
        try:
            response = completion(model=BACKUP_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=200)
            judge_res = parse_json_response(response.choices[0].message.content)
        except Exception as e:
            print(f"   ❌ [Error] Gemini ก็ล่มเหมือนกัน: {e}")

    if not judge_res:
        judge_res = {"score_correctness": 1, "judge_reason": "AI กรรมการทั้ง 2 ตัวไม่สามารถเชื่อมต่อได้"}

    return {
        "id": index, # <-- เพิ่มการเก็บลำดับข้อที่ตรงนี้
        "level": level,
        "question": question,
        "expected": expected,
        "answer": answer,
        "score_correctness": judge_res.get("score_correctness", 1),
        "judge_reason": judge_res.get("judge_reason", ""),
        "latency_sec": round(latency, 2)
    }

async def run_eval_pipeline():
    results = []
    print(f"🚀 เริ่มการประเมินผลระบบ Hybrid RAG (รวม 30 ข้อ)")
    
    for idx, case in enumerate(TEST_CASES, start=1):
        res = await evaluate_single_turn(case, idx)
        results.append(res)
        
        # [CRITICAL FIX] ป้องกัน API Rate Limit (หยุดพัก 4 วินาทีทุกข้อ)
        if idx < len(TEST_CASES):
            print("   ⏳ [Rate Limit Protection] พัก 4 วินาที เพื่อป้องกัน API โดนบล็อค...")
            await asyncio.sleep(4)
            
    df = pd.DataFrame(results)
    
    total_score = df['score_correctness'].sum()
    max_score = len(df) * 5
    percentage = (total_score / max_score) * 100
    
    if percentage >= 80:
        grade, color = "Excellent", "#22c55e"
    elif percentage >= 60:
        grade, color = "Good", "#eab308"
    else:
        grade, color = "Needs Improvement", "#ef4444"

    # [อัปเกรดหน้าตา HTML ใหม่] แก้ปัญหาจอทะลุขอบ และทำให้ดูสวยงามระดับ Pro
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>RAG Evaluation Report</title>
        <style>
            body {{ font-family: 'Sarabun', sans-serif; background-color: #f1f5f9; padding: 20px; color: #0f172a; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            .card {{ background: white; padding: 24px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); overflow-x: auto; }}
            h2 {{ margin-top: 0; color: #1e293b; }}
            
            /* บังคับตารางไม่ให้ทะลุขอบจอ */
            table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
            th, td {{ padding: 16px; border-bottom: 1px solid #e2e8f0; text-align: left; vertical-align: top; word-wrap: break-word; overflow-wrap: break-word; }}
            th {{ background-color: #1e293b; color: white; text-align: center; text-transform: uppercase; font-size: 14px; letter-spacing: 0.5px; }}
            
            /* ป้ายบอกระดับความยาก (Badges) */
            .badge-easy {{ background-color: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: bold; }}
            .badge-medium {{ background-color: #fef08a; color: #854d0e; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: bold; }}
            .badge-hard {{ background-color: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: bold; }}
            
            /* สีคะแนน */
            .score {{ font-weight: bold; padding: 6px 12px; border-radius: 6px; display: inline-block; text-align: center; width: 24px; font-size: 16px; }}
            .s-5 {{ background: #22c55e; color: white; }}
            .s-4 {{ background: #84cc16; color: white; }}
            .s-3 {{ background: #eab308; color: white; }}
            .s-2 {{ background: #f97316; color: white; }}
            .s-1 {{ background: #ef4444; color: white; }}
            
            /* ตกแต่ง Tag และรูปภาพ */
            .tag {{ font-family: monospace; background: #f1f5f9; border: 1px solid #cbd5e1; padding: 2px 6px; border-radius: 4px; font-size: 12px; color: #334155; word-break: break-all; }}
            img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0; margin-top: 8px; }}
            .reason {{ color: #475569; font-size: 13px; line-height: 1.5; }}
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&display=swap" rel="stylesheet">
    </head>
    <body>
        <div class="container">
            <div class="card" style="text-align: center;">
                <h2>📊 ผลประเมินระบบ Hybrid RAG</h2>
                <h1 style="font-size: 48px; margin: 10px 0; color: {color}">{percentage:.2f}% <span style="font-size: 24px;">({grade})</span></h1>
                <p style="color: #64748b;"><strong>เอกสารทดสอบ:</strong> {DOC_ID} | <strong>จำนวนคำถาม:</strong> 30 ข้อ</p>
            </div>
            <div class="card">
                <table>
                    <thead>
                        <tr>
                            <th width="5%">ข้อ</th>
                            <th width="10%">ระดับ</th>
                            <th width="25%">คำถาม</th>
                            <th width="35%">คำตอบจากระบบ</th>
                            <th width="5%">คะแนน</th>
                            <th width="20%">เหตุผลจาก AI</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    for _, row in df.iterrows():
        ans_display = str(row['answer']).replace("[SHOW_TABLE:", "<span class='tag'>[SHOW_TABLE:").replace("[SHOW_IMAGE:", "<span class='tag'>[SHOW_IMAGE:")
        ans_display = ans_display.replace("]", "]</span>")
        
        # จัดการคลาสของ Badge ให้ตรงกับระดับ
        badge_class = f"badge-{row['level'].lower()}"
        
        html += f"""
                        <tr>
                            <td style="text-align: center; font-weight: bold; font-size: 16px; color: #475569;">{row['id']}</td>
                            <td style="text-align: center;"><span class="{badge_class}">{row['level'].upper()}</span></td>
                            <td style="font-weight: 600; color: #1e293b;">{row['question']}</td>
                            <td style="font-size:14px; color:#334155; line-height: 1.6;">{ans_display}</td>
                            <td style="text-align: center;"><span class="score s-{row['score_correctness']}">{row['score_correctness']}</span></td>
                            <td class="reason">{row['judge_reason']}</td>
                        </tr>
        """
    html += "</tbody></table></div></div></body></html>"
    
    with open("eval_report_th.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✅ การประเมินเสร็จสิ้น! รีพอร์ตถูกบันทึกไว้ที่: {os.path.abspath('eval_report_th.html')}") 

if __name__ == "__main__":
    asyncio.run(run_eval_pipeline())