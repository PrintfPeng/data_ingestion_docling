#!/bin/bash
# =====================================================================
# upload_and_deploy.sh
# รันจาก Git Bash บน Windows เพื่ออัปโหลด + deploy บน server
# =====================================================================

SERVER="yruadmin@10.20.41.108"
REMOTE_DIR="/opt/ingestion-pipeline"

echo "=================================================="
echo " Upload & Deploy to $SERVER"
echo "=================================================="
echo ""

# Step 1: สร้างโฟลเดอร์บน server
echo "[1/3] สร้างโฟลเดอร์บน server..."
ssh -o StrictHostKeyChecking=no "$SERVER" \
  "sudo mkdir -p $REMOTE_DIR && sudo chown yruadmin:yruadmin $REMOTE_DIR && echo OK"

# Step 2: rsync ไฟล์ (ยกเว้น venv, cache, ข้อมูล local)
echo ""
echo "[2/3] อัปโหลดไฟล์..."
rsync -avz --progress \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'chroma_db/' \
  --exclude 'ingested/' \
  --exclude 'uploads/' \
  --exclude 'logs/' \
  --exclude '*.png' \
  --exclude 'venv/' \
  -e "ssh -o StrictHostKeyChecking=no" \
  ./ "$SERVER:$REMOTE_DIR/"

# Step 3: รัน docker-compose บน server
echo ""
echo "[3/3] Deploy ด้วย Docker Compose..."
ssh -o StrictHostKeyChecking=no "$SERVER" "
  cd $REMOTE_DIR
  echo '--- chmod .env ---'
  chmod 600 .env
  echo '--- docker compose pull nginx image ---'
  docker compose pull nginx 2>/dev/null || true
  echo '--- docker compose build ---'
  docker compose build --no-cache
  echo '--- docker compose up ---'
  docker compose up -d
  echo '--- รอ 10s แล้วตรวจ status ---'
  sleep 10
  docker compose ps
  echo ''
  echo '--- Health check ---'
  curl -s http://localhost:80/health || echo 'ยังไม่พร้อม (รอ backend start)'
"

echo ""
echo "=================================================="
echo "Done! เปิดที่: http://10.20.41.108/"
echo "ดู log: ssh $SERVER 'docker compose -f $REMOTE_DIR/docker-compose.yml logs -f'"
echo "=================================================="
