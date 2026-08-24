# 🚀 Deployment Guide — Ubuntu/CentOS Server + Docker + NVIDIA GPU

คู่มือ deploy `data_ingestion_docling` ขึ้น server มหาลัย

**สภาพแวดล้อมเป้าหมาย:**
- Ubuntu 22.04 / CentOS Stream 9 (มี sudo/root)
- Docker Engine + Docker Compose v2
- NVIDIA GPU + driver ล่าสุด
- Port 80 (และ 443 เมื่อพร้อม cert) เปิดให้ผู้ใช้เข้า

---

## 1) ติดตั้ง prerequisites บน host (ครั้งเดียว)

### 1.1 Docker Engine + Compose plugin

Ubuntu:
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
docker version && docker compose version
```

CentOS/RHEL:
```bash
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER && newgrp docker
```

### 1.2 NVIDIA Container Toolkit (บังคับสำหรับ GPU)

Ubuntu:
```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

ตรวจว่า container เห็น GPU:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```
ต้องเห็นตาราง GPU ปกติ

### 1.3 เปิด firewall ให้พอร์ต 80/443

Ubuntu (ufw):
```bash
sudo ufw allow 80/tcp && sudo ufw allow 443/tcp && sudo ufw reload
```

CentOS (firewalld):
```bash
sudo firewall-cmd --permanent --add-service=http --add-service=https
sudo firewall-cmd --reload
```

---

## 2) Clone repo และตั้ง environment

```bash
sudo mkdir -p /opt/ingestion && sudo chown -R $USER:$USER /opt/ingestion
cd /opt/ingestion
git clone https://github.com/PrintfPeng/data_ingestion_docling.git
cd data_ingestion_docling

cp .env.example .env
nano .env    # ← ใส่ API keys จริง (CUSTOM_API_KEY, GOOGLE_API_KEY, OCR_PASSWORD)
chmod 600 .env
```

โฟลเดอร์ persist จะถูกสร้างตอน compose up เอง (`ingested/`, `uploads/`, `chroma_db/`, `logs/`)

---

## 3) Build & start

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f backend
```

**Build ครั้งแรก:** 15–25 นาที (image ~7 GB จาก base pytorch/cuda)
**Start ครั้งแรก:** อีก 3–5 นาทีเพื่อ download embedding model `intfloat/multilingual-e5-large` (~2.2 GB) เข้า `hf_cache` volume — ครั้งต่อไปจะเร็ว

ทดสอบว่าติด:
```bash
curl http://localhost/health
# {"status":"ok","service":"backend","mode":"multi_doc",...}
```

เปิดเบราว์เซอร์เข้า `http://<server-ip>/` จะ redirect ไปหน้า chat UI

---

## 4) ตรวจว่า GPU ใช้งานจริง

```bash
docker compose exec backend python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

ควรได้ `CUDA: True | GPU: NVIDIA ...`

ถ้าอยากให้ embedding ใช้ GPU (default เป็น CPU) แก้ [backend/services/embeddings.py:47](backend/services/embeddings.py#L47) จาก `'cpu'` เป็น `'cuda'`

---

## 5) ตั้ง domain + HTTPS (ทำเมื่อได้ cert แล้ว)

**5.1 ขอ cert จากมหาลัย** หรือใช้ Let's Encrypt ผ่าน certbot standalone (ต้องหยุด nginx ชั่วคราว):
```bash
sudo apt install -y certbot
docker compose stop nginx
sudo certbot certonly --standalone -d ai-ingest.your-university.ac.th
sudo cp /etc/letsencrypt/live/ai-ingest.your-university.ac.th/fullchain.pem ./nginx/certs/
sudo cp /etc/letsencrypt/live/ai-ingest.your-university.ac.th/privkey.pem  ./nginx/certs/
sudo chown -R $USER:$USER ./nginx/certs
```

**5.2 เปิด HTTPS ใน [nginx/nginx.conf](nginx/nginx.conf):**
- Uncomment block `server { listen 443 ssl http2; ... }` ท้ายไฟล์
- ใส่ `server_name` เป็น domain จริง

**5.3 เปิดพอร์ต 443 ใน [docker-compose.yml](docker-compose.yml):**
- Uncomment `- "443:443"` ใต้ service `nginx`
- Uncomment `- ./nginx/certs:/etc/nginx/certs:ro` ใต้ volumes

**5.4 Restart:**
```bash
docker compose up -d nginx
```

---

## 6) Operations

**ดู log แบบ live:**
```bash
docker compose logs -f backend
docker compose logs -f nginx
```

**Restart หลังแก้ code:**
```bash
docker compose up -d --build backend
```

**สำรอง data:**
```bash
tar -czf backup-$(date +%F).tgz ingested/ chroma_db/ uploads/ logs/
```

**ล้าง Vector DB (เริ่มใหม่):**
```bash
docker compose down
rm -rf chroma_db/*
docker compose up -d
```

**หยุด/ลบทั้งหมด:**
```bash
docker compose down            # หยุดแต่เก็บ volume ไว้
docker compose down -v         # ลบ named volume ด้วย (จะโหลด HF model ใหม่)
```

---

## 7) Troubleshooting

| อาการ | สาเหตุที่พบบ่อย | วิธีตรวจ |
|-------|-------------------|----------|
| `nvidia-container-cli: initialization error` | ยังไม่ได้ติดตั้ง toolkit หรือลืม restart docker | รัน `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi` |
| `backend` ค้างที่ `Loading Local Embedding Model` | กำลัง download HF model (2 GB) ครั้งแรก | รอ 3–5 นาที ดู log `docker compose logs -f backend` |
| Upload PDF ใหญ่ error 413 | `client_max_body_size` เล็กเกิน | แก้ใน [nginx/nginx.conf](nginx/nginx.conf) แล้ว `docker compose restart nginx` |
| Ingestion pipeline ช้ามาก | `run_ingestion` เรียก OCR/LLM sync ทีละหน้า | เป็นพฤติกรรมปกติ; ดู log ว่าค้างที่ขั้นไหน |
| RAG ตอบว่า "ไม่ทราบ" ตลอด | Chroma ว่างเปล่า / ยังไม่มี doc ที่ ingest | `docker compose exec backend python -m scripts.inspect_db` |

---

## 8) ไฟล์ที่เกี่ยวข้อง (สรุป)

- [Dockerfile](Dockerfile) — image สำหรับ backend
- [docker-compose.yml](docker-compose.yml) — orchestration
- [nginx/nginx.conf](nginx/nginx.conf) — reverse proxy
- [.env.example](.env.example) — template environment
- [.dockerignore](.dockerignore) — ไฟล์ที่ไม่ copy เข้า image
