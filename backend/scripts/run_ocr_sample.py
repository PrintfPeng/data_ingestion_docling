from ingestion.ocr_extractor import ocr_extract_document

def main():
    pdf_path = "sample.pdf"  # หรือ path ที่ถูก
    result = ocr_extract_document(pdf_path)

    for page_info in result.texts:
        print("=== page", page_info["page"], "===")
        print(page_info["content"][:800])
        print()

if __name__ == "__main__":
    main()
