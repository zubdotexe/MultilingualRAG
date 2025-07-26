import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

def lazy_load_pdf_pages(file_path):
    doc = fitz.open(file_path)
    for page_number in range(len(doc)):
        yield doc.load_page(page_number)  # yields one page at a time
    doc.close()

# Store texts and corresponding page as metadata
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

# Splitting texts and creating documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)

documents = text_splitter.create_documents(texts, metadatas=metadatas)

print("len of texts", len(texts))
print("len of docs", len(documents))
print(documents[0].page_content)
print(documents[0].metadata)