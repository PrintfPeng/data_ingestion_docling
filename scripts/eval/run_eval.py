"""
scripts/eval/run_eval.py

Lightweight RAG evaluation — runs each golden Q against /ask endpoint,
computes 4 metrics per question, saves JSON + HTML report.

Metrics
-------
1. **keyword_recall** — fraction of expected_keywords found in answer
   (case-insensitive substring match, robust to word variations)
2. **doc_match** — 1.0 if any retrieved source's doc_id ∈ expected_doc_ids
3. **judge_score** — LLM judge (Gemini 2.5 Flash via OpenRouter) grades
   answer against ground_truth on a 0-1 scale
4. **latency_s** — seconds from POST to response

Usage
-----
  export APP_API_KEY=sk-ingest-...
  export VISION_API_KEY=sk-or-v1-...
  export BASE_URL=http://localhost:9999
  python scripts/eval/run_eval.py
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import html
import argparse
from pathlib import Path
from typing import List, Dict, Any

import requests
from openai import OpenAI


HERE = Path(__file__).resolve().parent
GOLDEN_FILE = HERE / "golden_qa.json"
REPORT_DIR = HERE / "reports"
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "google/gemini-2.5-flash")
BASE_URL = os.getenv("BASE_URL", "http://localhost:9999").rstrip("/")
APP_API_KEY = os.getenv("APP_API_KEY", "")
VISION_API_KEY = os.getenv("VISION_API_KEY", "")


# --------------------- Metrics ---------------------

def keyword_recall(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 1.0
    a = (answer or "").lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in a)
    return hits / len(expected_keywords)


def doc_match(sources: List[Dict], expected_doc_ids: List[str]) -> float:
    if not expected_doc_ids:
        return 1.0
    wanted = set(expected_doc_ids)
    for s in sources:
        if s.get("doc_id") in wanted:
            return 1.0
    return 0.0


def judge_score(question: str, ground_truth: str, answer: str) -> Dict[str, Any]:
    """LLM-as-judge: score answer 0-1 vs ground truth."""
    if not VISION_API_KEY:
        return {"score": None, "reason": "no VISION_API_KEY set"}

    client = OpenAI(
        api_key=VISION_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    prompt = f"""คุณคือผู้ประเมินคุณภาพคำตอบ RAG ประเมินว่าคำตอบของระบบตรงกับ Ground Truth หรือไม่

คำถาม: {question}
Ground Truth: {ground_truth}
คำตอบของระบบ: {answer}

ให้คะแนน 0-1 (ทศนิยม 2 ตำแหน่ง):
- 1.00 = ตอบถูกทั้งหมด ครอบคลุมข้อเท็จจริงหลักใน Ground Truth
- 0.70-0.99 = ตอบถูกเป็นส่วนใหญ่ อาจขาดรายละเอียดบางอย่าง
- 0.30-0.69 = ตอบบางส่วน หรือคลุมเครือ
- 0.01-0.29 = ตอบผิด/ไม่ตรงประเด็นเป็นส่วนใหญ่
- 0.00 = ตอบผิดสิ้นเชิง / ไม่รู้

ตอบเป็น JSON เท่านั้น: {{"score": 0.XX, "reason": "..."}}"""
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = resp.choices[0].message.content or ""
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return {"score": float(data.get("score", 0)), "reason": data.get("reason", "")}
        return {"score": None, "reason": f"parse-fail: {text[:200]}"}
    except Exception as e:
        return {"score": None, "reason": f"error: {e}"}


# --------------------- Runner ---------------------

def call_ask(question: str) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if APP_API_KEY:
        headers["Authorization"] = f"Bearer {APP_API_KEY}"
    payload = {"query": question, "top_k": 5, "mode": "auto"}
    t0 = time.time()
    r = requests.post(f"{BASE_URL}/ask", headers=headers, json=payload, timeout=300)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    return {
        "answer": data.get("answer", ""),
        "sources": data.get("sources", []),
        "latency_s": round(elapsed, 2),
    }


def run(golden: List[Dict]) -> List[Dict]:
    results = []
    for i, q in enumerate(golden, start=1):
        print(f"\n[{i}/{len(golden)}] {q['question']}")
        try:
            resp = call_ask(q["question"])
            answer = resp["answer"]
            sources = resp["sources"]
            latency = resp["latency_s"]
        except Exception as e:
            print(f"  ⚠️ ask failed: {e}")
            results.append({**q, "answer": "", "sources": [], "latency_s": None,
                            "keyword_recall": 0, "doc_match": 0,
                            "judge_score": 0, "judge_reason": f"ask-error: {e}"})
            continue

        kr = keyword_recall(answer, q.get("expected_keywords", []))
        dm = doc_match(sources, q.get("expected_doc_ids", []))
        judge = judge_score(q["question"], q["ground_truth"], answer)

        print(f"  ans: {answer[:120]}...")
        print(f"  keyword_recall={kr:.2f}  doc_match={dm:.2f}  judge={judge.get('score')}  latency={latency}s")

        results.append({
            **q,
            "answer": answer,
            "sources": [{"doc_id": s.get("doc_id"), "page": s.get("page"),
                         "content_snippet": (s.get("content") or "")[:150]}
                        for s in sources],
            "latency_s": latency,
            "keyword_recall": round(kr, 3),
            "doc_match": dm,
            "judge_score": judge.get("score"),
            "judge_reason": judge.get("reason", ""),
        })
    return results


def aggregate(results: List[Dict]) -> Dict[str, Any]:
    valid_judge = [r["judge_score"] for r in results if isinstance(r.get("judge_score"), (int, float))]
    latencies = [r["latency_s"] for r in results if isinstance(r.get("latency_s"), (int, float))]
    return {
        "n": len(results),
        "avg_keyword_recall": round(sum(r["keyword_recall"] for r in results) / max(len(results), 1), 3),
        "avg_doc_match": round(sum(r["doc_match"] for r in results) / max(len(results), 1), 3),
        "avg_judge_score": round(sum(valid_judge) / max(len(valid_judge), 1), 3) if valid_judge else None,
        "avg_latency_s": round(sum(latencies) / max(len(latencies), 1), 2) if latencies else None,
        "p95_latency_s": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2) if len(latencies) >= 2 else None,
    }


def render_html(results: List[Dict], summary: Dict[str, Any], meta: Dict[str, Any]) -> str:
    def cell_color(v, high_is_good=True):
        if v is None:
            return "#f1f5f9"
        if not isinstance(v, (int, float)):
            return "#f1f5f9"
        if high_is_good:
            if v >= 0.8:  return "#dcfce7"
            if v >= 0.5:  return "#fef9c3"
            return "#fee2e2"
        return "#f1f5f9"

    rows_html = []
    for r in results:
        rows_html.append(f"""
        <tr>
          <td>{html.escape(r['id'])}</td>
          <td>{html.escape(r['question'])}</td>
          <td>{html.escape(r['ground_truth'])}</td>
          <td class="answer">{html.escape((r.get('answer') or '')[:300])}{'…' if len(r.get('answer') or '')>300 else ''}</td>
          <td style="background:{cell_color(r['keyword_recall'])}">{r['keyword_recall']:.2f}</td>
          <td style="background:{cell_color(r['doc_match'])}">{r['doc_match']:.0f}</td>
          <td style="background:{cell_color(r['judge_score'])}">{r['judge_score'] if r['judge_score'] is not None else '—'}</td>
          <td>{r['latency_s'] if r['latency_s'] is not None else '—'}s</td>
        </tr>""")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>RAG Eval Report</title>
<style>
  body {{ font-family: 'Sarabun', system-ui, sans-serif; margin: 24px; color: #1e293b; background: #f8fafc; }}
  h1 {{ margin: 0 0 4px; }}
  .meta {{ color: #64748b; font-size: 13px; margin-bottom: 24px; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 24px 0; }}
  .card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; }}
  .card .label {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}
  .card .val {{ font-size: 28px; font-weight: 700; color: #0284c7; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); font-size: 13px; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; vertical-align: top; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  td.answer {{ max-width: 350px; }}
  td {{ font-family: 'Sarabun', sans-serif; }}
  .metric-header {{ font-size: 11px; }}
</style>
</head><body>
  <h1>🧪 RAG Evaluation Report</h1>
  <div class="meta">
    Timestamp: {meta['ts']} · N questions: {summary['n']} · Base URL: {meta['base_url']} · Judge: {meta['judge_model']}
  </div>
  <div class="summary">
    <div class="card"><div class="label">Avg Keyword Recall</div><div class="val">{summary['avg_keyword_recall']:.2f}</div></div>
    <div class="card"><div class="label">Avg Doc Match</div><div class="val">{summary['avg_doc_match']:.2f}</div></div>
    <div class="card"><div class="label">Avg Judge Score</div><div class="val">{summary['avg_judge_score'] if summary['avg_judge_score'] is not None else '—'}</div></div>
    <div class="card"><div class="label">Avg Latency</div><div class="val">{summary['avg_latency_s']}s</div></div>
    <div class="card"><div class="label">P95 Latency</div><div class="val">{summary['p95_latency_s']}s</div></div>
  </div>
  <table>
    <thead><tr>
      <th>ID</th><th>Question</th><th>Ground Truth</th><th>Answer</th>
      <th class="metric-header">Kw Recall</th><th class="metric-header">Doc</th>
      <th class="metric-header">Judge</th><th class="metric-header">Latency</th>
    </tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="baseline", help="Report tag (added to filename)")
    args = parser.parse_args()

    if not GOLDEN_FILE.exists():
        print(f"golden file not found: {GOLDEN_FILE}", file=sys.stderr)
        sys.exit(1)

    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(golden)} golden Q&A")
    print(f"Base URL: {BASE_URL}  ·  Judge: {JUDGE_MODEL}")
    print(f"APP_API_KEY set: {bool(APP_API_KEY)}  ·  VISION_API_KEY set: {bool(VISION_API_KEY)}")

    results = run(golden)
    summary = aggregate(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    meta = {"ts": ts, "base_url": BASE_URL, "judge_model": JUDGE_MODEL, "tag": args.tag}

    json_path = REPORT_DIR / f"eval_{args.tag}_{ts}.json"
    json_path.write_text(json.dumps({"meta": meta, "summary": summary, "results": results},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {json_path}")

    html_path = REPORT_DIR / f"eval_{args.tag}_{ts}.html"
    html_path.write_text(render_html(results, summary, meta), encoding="utf-8")
    print(f"Saved HTML: {html_path}")

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
