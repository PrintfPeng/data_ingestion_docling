# test_md.py
from pathlib import Path
import json
from ingestion.schema import IngestedDocument, DocumentMetadata, TextBlock, TableBlock, ImageBlock
from ingestion.markdown_generator import generate_markdown

doc_id = "operation_manual_sharp"
root_dir = Path("ingested") / doc_id

print(f"Checking folder: {root_dir.resolve()}")

if not root_dir.exists():
    print("❌ Folder not found!")
    exit()

# Load JSONs
try:
    with open(root_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(root_dir / "text.json", "r", encoding="utf-8") as f:
        texts = json.load(f)
    with open(root_dir / "table.json", "r", encoding="utf-8") as f:
        tables = json.load(f)
    with open(root_dir / "image.json", "r", encoding="utf-8") as f:
        images = json.load(f)

    # Reconstruct Object
    doc = IngestedDocument(
        metadata=DocumentMetadata(**meta),
        texts=[TextBlock(**t) for t in texts],
        tables=[TableBlock(**t) for t in tables],
        images=[ImageBlock(**t) for t in images]
    )

    # Generate MD
    generate_markdown(doc, root_dir)
    print("✅ Manually Generated Markdown Success!")

except Exception as e:
    print(f"❌ Error: {e}")