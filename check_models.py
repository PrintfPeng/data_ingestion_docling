import os
import requests
import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# เริ่มต้น Rich Console
console = Console()

load_dotenv()

def get_models():
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = "http://111.223.37.51/v1/models"
    
    if not api_key:
        console.print("[bold red]❌ Error: ไม่เจอ API Key ใน .env[/bold red]")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    console.print(f"[bold cyan]🔍 Connecting to:[/bold cyan] {base_url} ...")

    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            models_list = data.get("data", [])
            
            # --- สร้างตาราง ---
            table = Table(
                title=f"🤖 Available AI Models ({len(models_list)})",
                box=box.ROUNDED,
                header_style="bold yellow",
                border_style="blue"
            )

            # กำหนดคอลัมน์
            table.add_column("No.", justify="right", style="dim")
            table.add_column("Model ID (Name)", style="bold white")
            table.add_column("Owned By", style="green")
            table.add_column("Type", style="magenta")

            # เรียงชื่อโมเดลตามตัวอักษร
            models_list.sort(key=lambda x: x.get('id', ''))

            # วนลูปใส่ข้อมูล
            for idx, model in enumerate(models_list, 1):
                m_id = model.get('id', 'Unknown')
                m_owner = model.get('owned_by', '-')
                m_obj = model.get('object', 'model')
                
                table.add_row(str(idx), m_id, m_owner, m_obj)

            console.print(table)
            console.print(f"[bold green]✅ Success![/bold green] Found {len(models_list)} models.\n")
            
        else:
            console.print(f"[bold red]💥 API Error: {response.status_code}[/bold red]")
            console.print(Panel(response.text, title="Error Details", border_style="red"))

    except Exception as e:
        console.print(f"[bold red]💥 Connection Failed:[/bold red] {e}")

if __name__ == "__main__":
    get_models()