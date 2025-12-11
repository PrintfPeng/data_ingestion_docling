from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import json


# โฟลเดอร์เก็บ log = <root>/logs/qa_log.jsonl
ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "qa_log.jsonl"

LOG_DIR.mkdir(exist_ok=True)


def append_log(entry: Dict[str, Any]) -> None:
    """
    บันทึก 1 event ลงไฟล์แบบ JSONL (แถวละ 1 JSON)

    entry ควรมี key อย่างน้อย:
      - query, answer, doc_ids, intent, mode, sources
    """
    payload = dict(entry)
    payload.setdefault("ts", datetime.utcnow().isoformat() + "Z")

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """
    อ่าน log ย้อนหลังใหม่สุดไม่เกิน limit รายการ
    """
    if not LOG_FILE.exists():
        return []

    with LOG_FILE.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        return []

    selected = lines[-limit:]
    logs: List[Dict[str, Any]] = []
    for ln in selected:
        try:
            logs.append(json.loads(ln))
        except json.JSONDecodeError:
            continue

    return logs
