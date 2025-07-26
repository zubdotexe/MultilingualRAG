# 📄 PDF Semantic Search App

This is a semantic search application built with **LangChain**, **ChromaDB**, and **Streamlit**, allowing you to retrieve meaningful results from Bangla and English PDFs using powerful multilingual embeddings.


## 🚀 Features

- ✅ PDF text extraction with `PyMuPDF`
- ✅ Sentence embedding with [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)
- ✅ Vector storage using `Chroma`
- ✅ Fast, semantically relevant retrieval
- ✅ User-friendly interface with `Streamlit`
- ✅ Supports **Bangla** and **English** PDFs


## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-directory>  
```

### 2. Create Conda Virtual env and activate it
```bash
conda create -p ./venv python=3.11
conda activate ./venv
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run main.py
```

## Sample Queries
### 1. কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
```
অনুপমের ভাগ্য দেবতা হিসেবে মামাকে উল্লেখ করা হয়েছে।
```
### 2. বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
```
আমি জানি না।
```
### 3. অনুপমের বয়স কত?
```
অনুপমের বয়স সাতার।
```
### 4. Who is Kalyani?
```
Kalyani is the daughter of Nabababu and wife of Shambhunath Sen. She is a character who takes a vow to serve her mother-in-law devotedly after facing humiliation and disrespect from her maternal uncle. Her strong adherence to maternal duty and traditional values is highlighted in the text.
```

## Q/A
### 1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?
I have used pymupdf to extract texts from the provided PDF. I have tested pdfplumber as well, but both these give almost the same result. The challenge is many Banlga words are unreadable due to the unicode incompatibility.

---

### 2. What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?
I have chosen RecursiveCharacterTextSplitter as it reursively splits paragraphs, sentences and words. It is able to get words fully and does not cut a word. I have kept the chunk size 1000 and chunk overlap 140 and this combination gave the best result.

---

### 3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?
I have used intfloat/multilingual-e5-base, a sentence transformer model from Hugging Face. The reasons are:
- It is open-source
- It understands both Bangla and English
- Compared to other multilingual or Bangla embedding models, it embeds better which is really helpful for retrieval

---
### 4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?
I have chosen ChromaDB as it is open-source and it is much simpler than FAISS to set up. Also, the similarity method is "similarity" as I needed to keep as much data as possible for retrieval as the format of the texts were creating some issues. I have tried using a threshold and MMR technique but those could not do much.

---
### 5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?
As both the question and the document chunks are embedded using the same model, the embeddings are similar for the same type of texts. Now, if the query is vague or missing context, the LLM provide **prompt** will be really helpful to avoid hallucinations. Using Multi Query Retriever would be helpful in case of vague queries.

---
### 6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?
The results have been relevant mostly. Howerver, there are some queries that are getting different outputs in different iterations. To solve these issues we can:  
- Get Bangla texts with unicode compatibility
- Test the chunk size on those extracted texts
- Use OpenAI embedding model for much better embeddings