import fitz

def lazy_load_pdf_pages(file_path):
    doc = fitz.open(file_path)
    for page_number in range(len(doc)):
        yield doc.load_page(page_number)  # yields one page at a time
    doc.close()

# Store texts and corresponding metadata
texts = []
metadatas = []

for page_number, page in enumerate(lazy_load_pdf_pages("HSC26-Bangla1st-Paper.pdf"), start=1):
    # text = page.get_text("text").strip()
    text = page.get_text("text")
    print("page_number", page_number)
    print(text)
    
    if text:
        texts.append(text)
        metadatas.append({"page": page_number})

print(texts)
print(metadatas)