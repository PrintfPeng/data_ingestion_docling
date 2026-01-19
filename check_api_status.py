import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI

# โหลดค่าจาก .env
load_dotenv()

def check_custom_api():
    print("\n" + "="*50)
    print("🛠️  CUSTOM API DIAGNOSTIC TOOL")
    print("="*50)

    # 1. ตรวจสอบ Environment Variables
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("CUSTOM_API_BASE")
    model_name = os.getenv("CUSTOM_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")

    print(f"📍 Base URL : {base_url}")
    print(f"🔑 API Key  : {api_key[:5]}...{api_key[-3:] if api_key else 'None'}")
    print(f"🧠 Model    : {model_name}")

    if not api_key or not base_url:
        print("\n❌ CRITICAL ERROR: ไม่พบ CUSTOM_API_KEY หรือ CUSTOM_API_BASE ในไฟล์ .env")
        return

    # คำเตือนเรื่อง URL
    if "chat/completions" in base_url:
        print("\n⚠️  WARNING: Base URL ดูแปลกๆ ปกติ OpenAI Client ต้องการแค่ '/v1'")
        print("   (เช่น http://111.223.37.51/v1)")

    # 2. เริ่มทดสอบการเชื่อมต่อ
    print("\n🔄 Connecting to Server...", end=" ")
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=10.0 # ตั้ง Timeout 10 วินาที
        )
        print("✅ Client Initialized.")
    except Exception as e:
        print(f"\n❌ Client Init Failed: {e}")
        return

    # 3. ทดสอบเรียก List Models (เช็คว่า Server ตอบสนองไหม)
    print("🔄 Checking Server Reachability (List Models)...", end=" ")
    try:
        models = client.models.list()
        print("✅ OK")
        # เช็คว่ามีโมเดลที่เราจะใช้ไหม
        found = any(m.id == model_name for m in models.data)
        if found:
            print(f"   (Found target model: {model_name})")
        else:
            print(f"⚠️  Warning: ไม่เจอชื่อโมเดล '{model_name}' ในรายการ แต่จะลองเรียกดู")
            print(f"   (Available models: {[m.id for m in models.data]})")
            
    except Exception as e:
        print(f"\n❌ Connect Failed: {e}")
        print("   👉 ข้อแนะนำ: เช็ค VPN, Firewall หรือดูว่า Server ล่มหรือไม่")
        return

    # 4. ทดสอบความฉลาด (Chat Completion)
    print(f"🔄 Testing Chat Completion with '{model_name}'...", end=" ")
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "ตอบสั้นๆ: 1 + 1 เท่ากับเท่าไหร่?"}
            ],
            max_tokens=50,
            temperature=0.1
        )
        duration = time.time() - start_time
        answer = response.choices[0].message.content.strip()
        
        print(f"✅ Success! ({duration:.2f}s)")
        print("\n💬 AI Answer:")
        print("-" * 20)
        print(answer)
        print("-" * 20)
        
        print("\n🎉 สรุป: ระบบ API ของคุณพร้อมใช้งานแล้ว!")

    except Exception as e:
        print(f"\n❌ Chat Error: {e}")
        if "404" in str(e):
            print("   👉 เป็นไปได้ว่าชื่อ Model ผิด หรือ URL ผิด")
        elif "401" in str(e):
            print("   👉 API Key ผิด")
        else:
            print("   👉 Server อาจจะมีปัญหาภายใน หรือไม่รองรับ Chat Completions")

if __name__ == "__main__":
    check_custom_api()