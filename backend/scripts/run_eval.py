import asyncio
import pandas as pd
import mlflow
import os
import sys
import time
import re
import json
from pathlib import Path
from litellm import completion

# --- SETUP PATH & ENV ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# เรียกใช้ฟังก์ชันหลักของระบบคุณ
from backend.services.rag import answer_question

# --- CONFIGURATION ---
JUDGE_MODEL = "openai/qwen/qwen-2.5-72b-instruct"

if not os.getenv("CUSTOM_API_KEY"):
    print("🔴 ERROR: CUSTOM_API_KEY is not set.")
    sys.exit(1)

os.environ["OPENAI_API_KEY"] = os.getenv("CUSTOM_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("CUSTOM_API_BASE", "http://111.223.37.51/v1")

# --- 1. GOLDEN DATASET (3 Levels of Difficulty) ---
eval_questions = [
    # === LEVEL 1: EASY (Direct Lookup) - ถามตัวเลขตรงๆ ===
    {"level": "Easy", "question": "รัฐบาลมีรายได้นำส่งคลังในปีงบประมาณ 2568 ทั้งสิ้นจำนวนเท่าใด", "ground_truth": "2,821,730 ล้านบาท"},
    {"level": "Easy", "question": "ในปี 2568 รัฐบาลเบิกจ่ายเงินงบประมาณไปทั้งหมดเท่าไหร่", "ground_truth": "3,723,068 ล้านบาท"},
    {"level": "Easy", "question": "ดุลเงินงบประมาณปี 2568 ขาดดุลเท่าไหร่", "ground_truth": "ขาดดุล 901,338 ล้านบาท"},
    {"level": "Easy", "question": "ดุลเงินนอกงบประมาณปี 2568 ขาดดุลเท่าไหร่", "ground_truth": "ขาดดุล 111,288 ล้านบาท"},
    {"level": "Easy", "question": "ยอดกู้เงินเพื่อชดเชยการขาดดุลในปี 2568 คือเท่าใด", "ground_truth": "922,700 ล้านบาท"},
    {"level": "Easy", "question": "เงินคงคลัง ณ สิ้นเดือนกันยายน 2568 มีจำนวนเท่าใด", "ground_truth": "580,311 ล้านบาท"},
    {"level": "Easy", "question": "เอกสารนี้ออกโดยหน่วยงานใด", "ground_truth": "กระทรวงการคลัง"},
    {"level": "Easy", "question": "ฉบับที่ของเอกสารข่าวนี้คือเลขอะไร", "ground_truth": "ฉบับที่ 138/2568"},
    {"level": "Easy", "question": "วันที่ของเอกสารข่าวนี้คือวันที่เท่าไหร่", "ground_truth": "22 ตุลาคม 2568"},
    {"level": "Easy", "question": "เบอร์โทรศัพท์สำหรับติดต่อสอบถามข้อมูลคือเบอร์อะไร", "ground_truth": "0-2126-5800"},

    # === LEVEL 2: MEDIUM (Comparison) - เปรียบเทียบปีเก่า/ใหม่ ===
    {"level": "Medium", "question": "รายได้ปี 2568 เพิ่มขึ้นหรือลดลงจากปี 2567 เป็นจำนวนเงินเท่าใด", "ground_truth": "เพิ่มขึ้น 24,802 ล้านบาท"},
    {"level": "Medium", "question": "รายได้ปี 2568 คิดเป็นเปอร์เซ็นต์เปลี่ยนแปลงจากปีก่อนเท่าไหร่", "ground_truth": "ร้อยละ 0.9"},
    {"level": "Medium", "question": "รายจ่ายปี 2568 สูงกว่าปี 2567 อยู่เท่าไหร่", "ground_truth": "สูงกว่า 180,671 ล้านบาท"},
    {"level": "Medium", "question": "การขาดดุลเงินงบประมาณปี 2568 เพิ่มขึ้นจากปี 2567 เท่าไหร่", "ground_truth": "155,869 ล้านบาท"},
    {"level": "Medium", "question": "ยอดกู้เงินชดเชยขาดดุลปี 2568 มากกว่าปี 2567 อยู่กี่บาท", "ground_truth": "339,700 ล้านบาท (922,700 - 583,000)"},
    {"level": "Medium", "question": "เงินคงคลังปลายงวดปี 2568 ลดลงจากต้นงวดเท่าไหร่", "ground_truth": "ลดลง 89,926 ล้านบาท"},
    {"level": "Medium", "question": "ดุลเงินสดก่อนกู้ปี 2568 ขาดดุลมากกว่าปี 2567 เท่าไหร่", "ground_truth": "129,353 ล้านบาท"},
    {"level": "Medium", "question": "รายจ่ายปี 2567 มีจำนวนเท่าใด", "ground_truth": "3,542,397 ล้านบาท"},
    {"level": "Medium", "question": "ดุลเงินนอกงบประมาณปี 2567 เป็นอย่างไร (เกินดุล/ขาดดุล เท่าไหร่)", "ground_truth": "ขาดดุล 137,804 ล้านบาท"},
    {"level": "Medium", "question": "เงินคงคลังต้นงวดของปี 2568 (ณ 1 ต.ค. 67) คือเท่าไหร่", "ground_truth": "670,237 ล้านบาท"},

    # === LEVEL 3: HARD (Synthesis & Context) - ต้องเข้าใจความสัมพันธ์ของข้อมูล ===
    {"level": "Hard", "question": "สรุปภาพรวมฐานะการคลังปี 2568 ว่าเป็นอย่างไร (รายรับ รายจ่าย ดุลต่างๆ)", "ground_truth": "รายได้ 2.82 ล้านล้านบาท รายจ่าย 3.72 ล้านล้านบาท ขาดดุลเงินงบประมาณ 9.01 แสนล้านบาท และขาดดุลเงินสดหลังกู้ 8.99 หมื่นล้านบาท"},
    {"level": "Hard", "question": "ทำไมเงินคงคลังถึงลดลงในปี 2568 ทั้งที่มีการกู้เงินชดเชยแล้ว", "ground_truth": "เพราะดุลเงินสดก่อนกู้ขาดดุลสูงถึง 1,012,626 ล้านบาท ซึ่งมากกว่าเงินที่กู้มา (922,700 ล้านบาท) ทำให้ดุลเงินสดหลังกู้ยังคงติดลบ 89,926 ล้านบาท"},
    {"level": "Hard", "question": "อธิบายองค์ประกอบของดุลเงินสดก่อนกู้ ว่ามาจากยอดใดรวมกันบ้าง", "ground_truth": "มาจาก ดุลเงินงบประมาณ (ขาดดุล 901,338) รวมกับ ดุลเงินนอกงบประมาณ (ขาดดุล 111,288) รวมเป็น 1,012,626 ล้านบาท"},
    {"level": "Hard", "question": "รายจ่ายปี 2568 เพิ่มขึ้นในอัตราที่สูงกว่าหรือต่ำกว่าการเพิ่มขึ้นของรายได้", "ground_truth": "สูงกว่า (รายจ่ายเพิ่ม 5.1% ในขณะที่รายได้เพิ่มเพียง 0.9%)"},
    {"level": "Hard", "question": "ถ้าเปรียบเทียบยอดกู้เงินชดเชยขาดดุล ปีไหนรัฐบาลกู้เงินมากกว่ากันระหว่าง 2567 กับ 2568 และมากกว่ากันกี่เปอร์เซ็นต์", "ground_truth": "ปี 2568 กู้มากกว่า คิดเป็นเพิ่มขึ้นร้อยละ 58.3"},
    {"level": "Hard", "question": "ยอดเงินคงคลังปลายงวดของปี 2567 สอดคล้องกับยอดเงินคงคลังต้นงวดของปี 2568 หรือไม่ เป็นจำนวนเท่าใด", "ground_truth": "สอดคล้อง คือจำนวน 670,237 ล้านบาท"},
    {"level": "Hard", "question": "ตัวเลข 89,926 ล้านบาท ปรากฏอยู่ในรายการใดบ้างในเอกสาร", "ground_truth": "1. ดุลเงินสดหลังกู้ (ติดลบ) 2. เงินคงคลังที่ลดลงจากต้นงวด (ส่วนเปลี่ยนแปลง)"},
    {"level": "Hard", "question": "จากข้อมูล รายจ่ายรัฐบาลประกอบด้วยรายการย่อยอะไรบ้าง (ตามหมายเลขข้อย่อย)", "ground_truth": "2.1 เงินงบประมาณจ่ายปีปัจจุบัน และ 2.2 เงินงบประมาณจ่ายจากปีก่อน (เหลื่อมจ่าย)"},
    {"level": "Hard", "question": "แนวโน้มการขาดดุลของรัฐบาลแย่ลงหรือดีขึ้นเมื่อเทียบกับปีก่อน (ดูจากดุลเงินสดก่อนกู้)", "ground_truth": "แย่ลง (ขาดดุลเพิ่มขึ้นจาก 8.83 แสนล้าน เป็น 1.01 ล้านล้าน)"},
    {"level": "Hard", "question": "ยืนยันความถูกต้องของสมการ: เงินคงคลังปลายงวด = เงินคงคลังต้นงวด + ดุลเงินสดหลังกู้", "ground_truth": "ถูกต้อง (670,237 + (-89,926) = 580,311)"}
]

# --- HELPER: Safe RAG Call (Correct Integration) ---
async def safe_rag_call(query):
    """เรียก RAG System ของคุณอย่างถูกวิธี"""
    try:
        # [IMPORTANT] เพิ่ม top_k เพื่อให้ระบบเห็นข้อมูลกว้างขึ้นสำหรับข้อยาก
        # และระบุ doc_ids ให้ตรงกับชื่อไฟล์ที่คุณ ingest เข้าไป
        response = await answer_question(
            query=query, 
            doc_ids=['Ministry of Finance October67 September68'], 
            top_k=20,  # เพิ่มเป็น 20 เพื่อความชัวร์
            mode="auto"
        )
        return response
    except Exception as e:
        print(f"   ❌ Error calling RAG: {e}")
        return {"answer": f"Error: {e}", "sources": []}

def judge_with_qwen(question, answer, ground_truth, context):
    """กรรมการตัดสิน (ใช้ Regex แกะ JSON เพื่อความชัวร์)"""
    
    prompt = f"""
    You are an impartial judge evaluating a RAG system regarding Thai Financial Data.
    
    Query: {question}
    Ground Truth: {ground_truth}
    Actual Answer: {answer}
    Retrieved Context: {context}

    Criteria:
    1. Correctness (1-5): Does the Actual Answer match the numbers/facts in Ground Truth? (Allow minor formatting diffs like ',' or 'ล้านบาท')
    2. Faithfulness (1-5): Is the answer supported by the Retrieved Context provided above?

    Response Format (JSON ONLY):
    {{"correctness": <int>, "faithfulness": <int>, "reason": "<short comment>"}}
    """

    try:
        response = completion(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        
        # [FIX] ใช้ Regex ดึง JSON เผื่อโมเดลพูดเยอะ
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            return {"correctness": 0, "faithfulness": 0, "reason": "Judge Output Error (No JSON)"}

    except Exception as e:
        return {"correctness": 0, "faithfulness": 0, "reason": f"System Error: {e}"}

# --- MAIN PROCESS ---
async def main():
    print(f"🚀 Starting Evaluation Pipeline (Model: {JUDGE_MODEL})")
    eval_data = []

    # 1. RAG Inference Loop
    for i, item in enumerate(eval_questions):
        print(f"[{i+1}/{len(eval_questions)}] [{item['level']}] Asking: {item['question']}")
        
        # A. Call Your System
        rag_res = await safe_rag_call(item['question'])
        answer = rag_res.get("answer", "No answer")
        sources = rag_res.get("sources", [])
        
        # B. Prepare Context for Judge
        # ดึงเนื้อหาจริงๆ จาก Sources ส่งให้กรรมการดูด้วย
        contexts = []
        for src in sources:
            if src.get("source") == "table":
                # เอา HTML หรือ Markdown ของตารางมาโชว์
                table_content = src.get("metadata", {}).get("markdown_content", "") or str(src)
                contexts.append(f"[Table Content]: {table_content[:800]}") 
            else:
                text_content = src.get("content") or src.get("metadata", {}).get("content", "")
                contexts.append(f"[Text Content]: {text_content[:500]}")
        
        full_context = "\n\n".join(contexts) if contexts else "No context retrieved."

        # C. Judge
        score = judge_with_qwen(item['question'], answer, item['ground_truth'], full_context)
        
        eval_data.append({
            "level": item['level'],
            "question": item['question'],
            "ground_truth": item['ground_truth'],
            "answer": answer,
            "score_correctness": score.get("correctness", 0),
            "score_faithfulness": score.get("faithfulness", 0),
            "judge_reason": score.get("reason", "")
        })
        
        print(f"   👉 Answer: {answer[:100]}...")
        print(f"   ✅ Score: {score.get('correctness')}/5\n")
        # time.sleep(1) # พักนิดหน่อยถ้าระบบโหลดหนัก

    # 2. Statistics
    df = pd.DataFrame(eval_data)
    
    # Calculate Scores by Level
    summary = df.groupby("level")["score_correctness"].mean()
    total_avg = df["score_correctness"].mean()
    percentage = (total_avg / 5.0) * 100
    
    grade, color = ("Poor 🔴", "red")
    if percentage >= 80: grade, color = ("Excellent 🟢", "green")
    elif percentage >= 70: grade, color = ("Good 🔵", "blue")
    elif percentage >= 50: grade, color = ("Fair 🟠", "orange")

    # 3. Print Summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY (3 Levels)")
    print("="*60)
    print(f"🔹 Easy   Avg: {summary.get('Easy', 0):.2f} / 5.0")
    print(f"🔸 Medium Avg: {summary.get('Medium', 0):.2f} / 5.0")
    print(f"🔥 Hard   Avg: {summary.get('Hard', 0):.2f} / 5.0")
    print("-" * 30)
    print(f"📈 Total Accuracy : {percentage:.2f}%")
    print(f"🏆 Verdict        : {grade}")
    print("="*60)

    # 4. HTML Report
    html = f"""
    <!DOCTYPE html>
    <html lang="th">
    <head>
        <meta charset="UTF-8">
        <title>RAG Evaluation Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background: #f4f6f9; }}
            .card {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; vertical-align: top; }}
            th {{ background: #2c3e50; color: white; }}
            .score {{ font-weight: bold; padding: 4px 8px; border-radius: 4px; color: white; display: inline-block; }}
            .s-5, .s-4 {{ background: green; }} .s-3 {{ background: orange; }} .s-2, .s-1, .s-0 {{ background: red; }}
            .Easy {{ border-left: 5px solid green; }}
            .Medium {{ border-left: 5px solid orange; }}
            .Hard {{ border-left: 5px solid red; }}
        </style>
    </head>
    <body>
        <h1>📊 RAG Evaluation Report</h1>
        <div class="card" style="text-align: center;">
            <h2>Total Score: <span style="color:{color}">{percentage:.2f}%</span> ({grade})</h2>
            <p>Easy: {summary.get('Easy', 0):.2f}/5 | Medium: {summary.get('Medium', 0):.2f}/5 | Hard: {summary.get('Hard', 0):.2f}/5</p>
        </div>
        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th width="5%">Level</th>
                        <th width="25%">Question</th>
                        <th width="20%">Ground Truth</th>
                        <th width="25%">AI Answer</th>
                        <th width="5%">Score</th>
                        <th width="20%">Reason</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for _, row in df.iterrows():
        html += f"""
                <tr class="{row['level']}">
                    <td><strong>{row['level']}</strong></td>
                    <td>{row['question']}</td>
                    <td style="color:#555">{row['ground_truth']}</td>
                    <td>{row['answer']}</td>
                    <td><span class="score s-{row['score_correctness']}">{row['score_correctness']}</span></td>
                    <td style="font-size:0.9em; color:gray;">{row['judge_reason']}</td>
                </tr>
        """
    
    html += "</tbody></table></div></body></html>"
    
    with open("eval_report.html", "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"💾 Report saved to: {os.path.abspath('eval_report.html')}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")