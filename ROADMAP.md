# 🗺️ Deployment Roadmap

## Phase 1 — Prototype (ตอนนี้)

**เป้าหมาย:** deploy ให้คุณคนเดียวใช้บน server มหาลัย 10.20.41.108 ผ่าน OpenRouter ($10 credit)

**Status:** ✅ พร้อม deploy — ทุกไฟล์ที่ต้องการมีครบแล้ว

**Stack:**
- Backend: FastAPI + uvicorn (single worker)
- LLM: OpenRouter (Qwen 2.5-72B)
- Vector DB: Chroma (local file)
- Reverse proxy: nginx (HTTP)
- Container: Docker Compose + GPU

**ขั้นตอน:**
1. เช็ค server spec (SSH เข้าไปรัน `uname -a; free -h; nvidia-smi`)
2. ติดตั้ง Docker + nvidia-container-toolkit ตาม [DEPLOY.md](DEPLOY.md)
3. `cp .env.example .env` แล้วใส่ OpenRouter key
4. `docker compose up -d`

**Limitations ที่ยอมรับใน Phase 1:**
- ไม่มี authentication (ใครเข้า IP ได้ก็ใช้ได้)
- HTTP only (ไม่มี HTTPS)
- Single key ทั้งระบบ (ทุก request ใช้ key เดียวกันของคุณ)
- ไม่มี rate limit
- ไม่มี usage tracking

---

## Phase 2 — API สำหรับนักศึกษา (อนาคต)

**เป้าหมาย:** เปิดให้นักศึกษาหลายคนใช้ผ่าน API ได้ พร้อมควบคุมค่าใช้จ่าย

### สิ่งที่ต้องเพิ่ม

#### 🔐 1. Authentication + Authorization
- **ตัวเลือก A: API key ของระบบเราเอง** (ไม่ใช่ OpenRouter key)
  - สร้าง endpoint `/register` + `/login` สำหรับนักศึกษา
  - ระบบออก `sk-univ-xxxxxxxx` ให้นักศึกษาเก็บ
  - Backend map `sk-univ-xxx` → user_id → ใช้ OpenRouter key ของ project (คุณจ่าย)
- **ตัวเลือก B: มหาลัย SSO**
  - เชื่อม LDAP / Microsoft Entra / Google Workspace
  - นักศึกษา login ด้วย email มหาลัย
- **ตัวเลือก C: Bring-your-own-key**
  - นักศึกษาสมัคร OpenRouter เอง แล้วเอา key มาใส่ผ่าน UI
  - คุณไม่ต้องจ่ายเงินเลย แต่ต้อง refactor code รับ key ต่อ request

#### 💰 2. Usage tracking + Rate limit
- ตาราง SQL เก็บ: user_id, timestamp, endpoint, tokens_in, tokens_out, cost
- Middleware นับ token ทุก call → เก็บลง DB
- Rate limit ต่อ user ต่อวัน (เช่น 100 requests/day, 500K tokens/day)
- Dashboard สรุปยอดใช้งาน + ค่าใช้จ่าย

#### 🌐 3. Domain + HTTPS
- ขอ subdomain กับ IT (`ai-ingest.your-univ.ac.th`)
- ขอ cert จากมหาลัย หรือ Let's Encrypt (ต้องเปิด public 80/443)
- Uncomment HTTPS block ใน [nginx/nginx.conf](nginx/nginx.conf) + [docker-compose.yml](docker-compose.yml)

#### 📊 4. API Documentation
- FastAPI ทำ OpenAPI ให้อยู่แล้วที่ `/docs` (Swagger UI)
- เพิ่ม examples + auth section
- อาจสร้าง Postman collection ให้นักศึกษา

#### 🚦 5. Multi-tenant refactor
Files ที่ต้องแก้ให้รับ `api_key` เป็น parameter (แทนที่จะดึงจาก env):
- [backend/services/rag.py](backend/services/rag.py) — `answer_question(api_key=...)`
- [ingestion/table_extractor.py](ingestion/table_extractor.py) — LLM/Vision clients
- [ingestion/image_extractor.py](ingestion/image_extractor.py) — captioning
- [ingestion/semantic_enricher.py](ingestion/semantic_enricher.py) — tag_sections
- [ingestion/document_classifier.py](ingestion/document_classifier.py)
- [ingestion/cleaner.py](ingestion/cleaner.py)
- [scripts/run_ingestion.py](scripts/run_ingestion.py) — OCR correction

**หรือ**: ใช้ **ContextVar** เก็บ key ต่อ request ก็ได้ (แก้น้อยกว่า, thread-safe)

#### 🗂️ 6. Data isolation
- ทุก doc/chunk ต้อง tag `user_id` ใน Chroma metadata
- Search filter ด้วย `user_id` เพื่อไม่ให้เห็นเอกสารคนอื่น
- Upload path แยก folder ต่อ user: `ingested/<user_id>/<doc_id>/`

#### 🛡️ 7. Security hardening
- CORS whitelist (ไม่ให้เว็บอื่น embed API)
- Input validation ทุก endpoint (max size, allowed file type)
- Sandbox subprocess ตอนรัน ingestion (จำกัด CPU/memory)
- Prompt injection defense (โดยเฉพาะเวลา user ส่ง query เข้า LLM)

### Timeline โดยประมาณ

| Sprint | งาน | ระยะเวลา |
|--------|-----|---------|
| **Phase 1** | Deploy prototype ใช้เอง | 1–2 วัน |
| Sprint 1 | Auth (API key ระบบเรา) + user table | 3–5 วัน |
| Sprint 2 | Usage tracking + rate limit | 2–3 วัน |
| Sprint 3 | Multi-tenant refactor + data isolation | 5–7 วัน |
| Sprint 4 | Domain + HTTPS + docs | 2–3 วัน |
| Sprint 5 | Testing, monitoring, hardening | 3–5 วัน |
| **Phase 2 total** | | **~4 สัปดาห์** (part-time) |

---

## Phase 3 — Production (ถ้า Phase 2 สำเร็จ)

- Prometheus + Grafana monitoring
- Loki centralized logs
- Backup automation (Chroma + uploads → S3/มหาลัย storage)
- CI/CD pipeline (GitHub Actions → SSH deploy)
- Load balancer (ถ้ามีผู้ใช้เยอะ)
- Redis cache สำหรับ RAG hit ซ้ำ
