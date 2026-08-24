import asyncio
import pandas as pd
import os
import sys
import time
import re
import json
from pathlib import Path
from litellm import completion

# บังคับ stdout ให้เป็น UTF-8 เพื่อไม่ให้ emoji ทำ Windows terminal (cp1252) พัง
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, Exception):
    pass

# --- SETUP PATH & ENV ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from backend.services.rag import answer_question

# --- CONFIGURATION (Hybrid System) ---
# 1. Primary Model (Qwen)
PRIMARY_MODEL = "openai/qwen/qwen-2.5-72b-instruct"
PRIMARY_PROVIDER = "custom"

# 2. Backup Model (Gemini)
BACKUP_MODEL = "gemini/gemini-2.5-flash"
BACKUP_PROVIDER = "google"

# --- CHECK API KEYS (moved to _check_api_keys() so this module can be imported safely) ---
HAS_PRIMARY = False
HAS_BACKUP = False


def _check_api_keys() -> None:
    """
    ตรวจสอบและตั้งค่า API keys — เรียกจาก main() เท่านั้น
    ถ้าไม่มี key เลยจะ sys.exit(1)
    """
    global HAS_PRIMARY, HAS_BACKUP

    # Setup Primary (Custom Qwen)
    if os.getenv("CUSTOM_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("CUSTOM_API_KEY")
        os.environ["OPENAI_API_BASE"] = os.getenv("CUSTOM_API_BASE", "http://111.223.37.51/v1")
        HAS_PRIMARY = True
        print(f"🔵 Primary Judge Ready: {PRIMARY_MODEL}")
    else:
        print("⚠️ Warning: CUSTOM_API_KEY missing. Primary judge unavailable.")

    # Setup Backup (Google Gemini)
    if os.getenv("GOOGLE_API_KEY"):
        HAS_BACKUP = True
        print(f"🟢 Backup Judge Ready: {BACKUP_MODEL}")
    else:
        print("⚠️ Warning: GOOGLE_API_KEY missing. Backup judge unavailable.")

    if not HAS_PRIMARY and not HAS_BACKUP:
        print("🔴 ERROR: No API Keys found (Neither Custom nor Google). Exiting.")
        sys.exit(1)

# --- QUESTION DATASET (30 Questions) ---
eval_questions = [
    # === LEVEL 1: EASY ===
    {"level": "Easy", "question": "สารทำความเย็นที่ใช้ในตู้เย็นรุ่นนี้คือชนิดใด", "ground_truth": "HFC-134a"},
    {"level": "Easy", "question": "ปริมาณสารทำความเย็นที่ระบุไว้ในข้อมูลจำเพาะคือเท่าใด", "ground_truth": "75 กรัม"},
    {"level": "Easy", "question": "แรงดันไฟฟ้าและความถี่ที่ระบุคือเท่าไหร่", "ground_truth": "220 โวลต์, 50 เฮิรตซ์"},
    {"level": "Easy", "question": "ฉนวนกันความร้อนที่ใช้เป็นชนิดใด", "ground_truth": "CYCLO PENTANE (NON CFC 100%)"},
    {"level": "Easy", "question": "อุปกรณ์หมายเลข 6 ในแผนภาพส่วนประกอบคือปุ่มอะไร", "ground_truth": "ปุ่มกดทำน้ำแข็งด่วน (Express Ice Making)"},
    {"level": "Easy", "question": "อุปกรณ์หมายเลข 7 ในแผนภาพคืออะไร", "ground_truth": "หลอดไฟ LED"},
    {"level": "Easy", "question": "อุปกรณ์หมายเลข 15 ที่ประตูตู้เย็นคืออะไร", "ground_truth": "ชั้นวางไข่"},
    {"level": "Easy", "question": "รุ่น SJ-C19E มีความกว้างเท่าใด", "ground_truth": "535 มม."},
    {"level": "Easy", "question": "ตู้เย็นรุ่น SJ-CP190E-CH มีปริมาตรความจุจริงกี่คิว (cu.ft)", "ground_truth": "5.9 คิว (167 ลิตร)"},
    {"level": "Easy", "question": "บริษัทผู้ผลิตขอขอบคุณที่เลือกใช้ผลิตภัณฑ์ยี่ห้อใด", "ground_truth": "ชาร์ป (SHARP)"},

    # === LEVEL 2: MEDIUM ===
    {"level": "Medium", "question": "หากต้องการเริ่มโหมดทำน้ำแข็งด่วน (Express Ice Making) ต้องทำอย่างไร", "ground_truth": "กดปุ่มทำน้ำแข็งด่วน (ปุ่มหมายเลข 6) จนไฟกระพริบ"},
    {"level": "Medium", "question": "โหมดทำน้ำแข็งด่วนจะหยุดทำงานอัตโนมัติภายในกี่ชั่วโมง", "ground_truth": "ภายใน 2 ชั่วโมง"},
    {"level": "Medium", "question": "ข้อห้ามในการใช้น้ำยาทำความสะอาดตู้เย็นมีอะไรบ้าง", "ground_truth": "ห้ามใช้น้ำร้อน, ผงขัดเงา, เบนซิน, ทินเนอร์, แอลกอฮอล์ หรือน้ำมันปิโตรเลียม"},
    {"level": "Medium", "question": "หากขอบยางประตูสกปรก ควรทำความสะอาดอย่างไร", "ground_truth": "ใช้แปรงสีฟันจุ่มน้ำอุ่นผสมน้ำยาล้างจานเช็ดถู แล้วเช็ดด้วยน้ำเปล่าให้สะอาด"},
    {"level": "Medium", "question": "ควรหลีกเลี่ยงการวางตู้เย็นในตำแหน่งใดบ้างเพื่อประหยัดพลังงาน", "ground_truth": "หลีกเลี่ยงการวางใกล้แหล่งกำเนิดความร้อน (เช่น เตาอบ) หรือกลางแสงแดด"},
    {"level": "Medium", "question": "ถ้าอาหารมีกลิ่นแรง ควรจัดการอย่างไรก่อนนำเข้าตู้เย็น", "ground_truth": "ควรห่อหุ้มอาหารให้มิดชิด (Wrapping is required) เพราะตัวดูดกลิ่นอาจกำจัดได้ไม่หมด"},
    {"level": "Medium", "question": "หลังจากเคลื่อนย้ายตู้เย็น ควรทิ้งช่วงเวลาอย่างน้อยกี่ชั่วโมงก่อนเสียบปลั๊ก", "ground_truth": "อย่างน้อย 2 ชั่วโมง (หรือตามที่คู่มือระบุเพื่อให้ระบบน้ำยาคงที่)"},
    {"level": "Medium", "question": "การปรับอุณหภูมิช่องแช่เย็นให้เย็นน้อยลง ต้องปรับปุ่มควบคุมไปทางตัวเลขใด (มากหรือน้อย)", "ground_truth": "ปรับไปทางตัวเลขน้อย (MIN)"},
    {"level": "Medium", "question": "ถ้าต้องการถอดชั้นวางของแบบเลื่อน (Slide tray) ต้องทำขั้นตอนใดบ้าง", "ground_truth": "ยกชั้นวางขึ้นเล็กน้อยแล้วดึงออกมาตรงๆ"},
    {"level": "Medium", "question": "ข้อแนะนำเพิ่มเติมเพื่อลดผลกระทบต่อสิ่งแวดล้อมเกี่ยวกับการเปิดประตูตู้เย็นคืออะไร", "ground_truth": "ไม่ควรเปิดประตูตู้เย็นบ่อย หรือเปิดทิ้งไว้นานเกินความจำเป็น"},

    # === LEVEL 3: HARD ===
    {"level": "Hard", "question": "ทำไมไฟ LED ในช่องแช่เย็นถึงกระพริบ และต้องเรียกช่างหรือไม่", "ground_truth": "ไฟกระพริบแสดงสถานะโหมดทำน้ำแข็งด่วน (Express Ice Making) ทำงานอยู่ ไม่ต้องเรียกช่าง เพราะเป็นอาการปกติ"},
    {"level": "Hard", "question": "หากพบว่าด้านข้างของตู้เย็นมีความร้อนเกิดขึ้น ถือว่าผิดปกติหรือไม่ เพราะเหตุใด", "ground_truth": "ไม่ผิดปกติ เพราะมีการฝังท่อระบายความร้อนไว้ที่ผนังตู้เพื่อป้องกันการเกิดหยดน้ำเกาะ (เหงื่อ)"},
    {"level": "Hard", "question": "ถ้าอาหารในช่องแช่เย็นเป็นน้ำแข็ง (Freeze) ทั้งที่ไม่ได้ปรับอุณหภูมิสูงสุด เกิดจากสาเหตุใดได้บ้าง", "ground_truth": "อาจเกิดจากอุณหภูมิภายนอกต่ำเกินไป (เช่น ฤดูหนาว) หรือวางอาหารใกล้มุมที่มีความเย็นจัด"},
    {"level": "Hard", "question": "เสียงดังคล้ายน้ำไหลจ๊อกๆ ในตู้เย็นเกิดจากอะไร", "ground_truth": "เกิดจากการไหลเวียนของสารทำความเย็นภายในระบบ ถือเป็นอาการปกติ"},
    {"level": "Hard", "question": "ทำไมถึงห้ามใช้น้ำร้อนทำความสะอาดชิ้นส่วนพลาสติกภายในตู้เย็น", "ground_truth": "เพราะความร้อนอาจทำให้ชิ้นส่วนพลาสติกผิดรูปหรือแตกร้าวได้"},
    {"level": "Hard", "question": "หากไฟดับเป็นเวลานาน ควรปฏิบัติต่ออาหารในตู้เย็นอย่างไร", "ground_truth": "ควรเปิดประตูตู้เย็นให้น้อยที่สุด เพื่อรักษาความเย็นภายในตู้ให้คงอยู่นานที่สุด"},
    {"level": "Hard", "question": "สาร Cyclo Pentane ที่ใช้เป็นฉนวน มีความเป็นมิตรต่อสิ่งแวดล้อมอย่างไร", "ground_truth": "เป็นสาร Non-CFC 100% ไม่ทำลายชั้นโอโซน"},
    {"level": "Hard", "question": "ระบบกำจัดกลิ่น (Deodorizer) สามารถกำจัดกลิ่นได้ทั้งหมดหรือไม่", "ground_truth": "ไม่ได้ทั้งหมด อาหารที่มีกลิ่นแรงยังคงต้องห่อหุ้มให้มิดชิด"},
    {"level": "Hard", "question": "หากตู้เย็นไม่เย็นเลย ควรตรวจสอบจุดใดเป็นอันดับแรก", "ground_truth": "ตรวจสอบว่าเสียบปลั๊กแน่นหรือไม่ หรือฟิวส์/เบรกเกอร์ของบ้านตัดหรือไม่"},
    {"level": "Hard", "question": "ทำไมไม่ควรวางสิ่งของหนักหรืออันตรายไว้บนหลังตู้เย็น", "ground_truth": "เพราะเมื่อเปิดปิดประตู แรงสั่นสะเทือนอาจทำให้ของตกลงมาเกิดอันตรายได้"}
]

# --- HELPER: Judge with Fallback Logic ---
def call_judge_api(model, prompt):
    """Helper function to call LiteLLM"""
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response.choices[0].message.content
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    else:
        raise ValueError("JSON not found in response")

def judge_answer_hybrid(question, answer, ground_truth, context):
    prompt = f"""
    You are an impartial judge evaluating a RAG system.
    
    Query: {question}
    Ground Truth: {ground_truth}
    Actual Answer: {answer}
    Retrieved Context: {context}

    Criteria:
    1. Correctness (1-5): Does the Actual Answer match the Ground Truth?
    2. Faithfulness (1-5): Is the answer supported by the Context?

    Response Format (JSON ONLY):
    {{"correctness": <int>, "faithfulness": <int>, "reason": "<Short reason in Thai>"}}
    """
    
    # 1. Try Primary (Qwen)
    if HAS_PRIMARY:
        try:
            # print("   🔹 Using Primary Judge (Qwen)...")
            return call_judge_api(PRIMARY_MODEL, prompt)
        except Exception as e:
            print(f"   ⚠️ Primary Judge Failed: {e}")
            if not HAS_BACKUP:
                return {"correctness": 0, "faithfulness": 0, "reason": "Primary Failed & No Backup"}
    
    # 2. Try Backup (Gemini) if Primary failed or unavailable
    if HAS_BACKUP:
        try:
            print("   🔸 Switching to Backup Judge (Gemini)...")
            time.sleep(2) # รอแป๊บนึงกัน Google Rate Limit
            return call_judge_api(BACKUP_MODEL, prompt)
        except Exception as e:
            print(f"   ❌ Backup Judge Failed: {e}")
            return {"correctness": 0, "faithfulness": 0, "reason": "Both Judges Failed"}

    return {"correctness": 0, "faithfulness": 0, "reason": "No Judges Available"}

async def safe_rag_call(query):
    try:
        response = await answer_question(
            query=query, 
            doc_ids=['operation_manual_sharp'], 
            top_k=10, 
            mode="auto"
        )
        return response
    except Exception as e:
        return {"answer": f"Error: {e}", "sources": []}

# --- MAIN ---
async def main():
    # ตรวจสอบ API key ก่อนเริ่ม (ย้ายมาจาก top-level เพื่อให้ import ไฟล์นี้ได้)
    _check_api_keys()

    print(f"🚀 Starting Hybrid Evaluation")
    if HAS_PRIMARY: print(f"   Main: {PRIMARY_MODEL}")
    if HAS_BACKUP:  print(f"   Backup: {BACKUP_MODEL}")
    print("-" * 50)

    eval_data = []

    for i, item in enumerate(eval_questions):
        print(f"[{i+1}/{len(eval_questions)}] [{item['level']}] Q: {item['question']}")
        
        # 1. RAG Call
        rag_res = await safe_rag_call(item['question'])
        answer = rag_res.get("answer", "No answer")
        sources = rag_res.get("sources", [])
        
        contexts = []
        for src in sources:
            text = src.get("content") or src.get("metadata", {}).get("content", "")
            contexts.append(text[:500])
        full_context = "\n---\n".join(contexts) if contexts else "No context."

        # 2. Hybrid Judge
        score = judge_answer_hybrid(item['question'], answer, item['ground_truth'], full_context)
        
        eval_data.append({
            "level": item['level'],
            "question": item['question'],
            "ground_truth": item['ground_truth'],
            "answer": answer,
            "score_correctness": score.get("correctness", 0),
            "score_faithfulness": score.get("faithfulness", 0),
            "judge_reason": score.get("reason", "")
        })
        
        print(f"   👉 AI: {answer[:60]}...")
        print(f"   ✅ Score: {score.get('correctness')}/5 ({score.get('reason')})\n")
        
        # พักนิดหน่อย
        time.sleep(1) 

    # 3. Report Generation
    df = pd.DataFrame(eval_data)
    total_avg = df["score_correctness"].mean()
    percentage = (total_avg / 5.0) * 100
    
    grade, color = "ปรับปรุง 🔴", "red"
    if percentage >= 80: grade, color = "ยอดเยี่ยม 🟢", "green"
    elif percentage >= 70: grade, color = "ดี 🔵", "blue"
    elif percentage >= 50: grade, color = "พอใช้ 🟠", "orange"

    print(f"\n📈 Final Score: {percentage:.2f}% ({grade})")

    html = f"""
    <!DOCTYPE html>
    <html lang="th">
    <head>
        <meta charset="UTF-8">
        <title>รายงานผลการทดสอบ RAG (Hybrid Judge)</title>
        <style>
            body {{ font-family: 'Sarabun', sans-serif; margin: 20px; background: #f4f6f9; }}
            .card {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; vertical-align: top; }}
            th {{ background: #2c3e50; color: white; }}
            .score {{ font-weight: bold; padding: 4px 8px; border-radius: 4px; color: white; display: inline-block; }}
            .s-5, .s-4 {{ background: #2ecc71; }} .s-3 {{ background: #f1c40f; color: black; }} .s-2, .s-1, .s-0 {{ background: #e74c3c; }}
            .Easy {{ border-left: 5px solid #2ecc71; }}
            .Medium {{ border-left: 5px solid #f1c40f; }}
            .Hard {{ border-left: 5px solid #e74c3c; }}
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap" rel="stylesheet">
    </head>
    <body>
        <div class="card" style="text-align: center;">
            <h2>คะแนนรวม: <span style="color:{color}">{percentage:.2f}%</span> ({grade})</h2>
            <p>Judge Model: Hybrid (Qwen Primary / Gemini Backup)</p>
        </div>
        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th width="10%">ระดับ</th>
                        <th width="30%">คำถาม</th>
                        <th width="30%">คำตอบ AI</th>
                        <th width="5%">คะแนน</th>
                        <th width="25%">เหตุผล</th>
                    </tr>
                </thead>
                <tbody>
    """
    for _, row in df.iterrows():
        html += f"""
                <tr class="{row['level']}">
                    <td><strong>{row['level']}</strong></td>
                    <td>{row['question']}</td>
                    <td>{row['answer']}</td>
                    <td><span class="score s-{row['score_correctness']}">{row['score_correctness']}</span></td>
                    <td style="color:gray;">{row['judge_reason']}</td>
                </tr>
        """
    html += "</tbody></table></div></body></html>"
    
    with open("eval_report_th.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"💾 Report saved to: {os.path.abspath('eval_report_th.html')}")

if __name__ == "__main__":
    asyncio.run(main())