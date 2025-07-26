import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Check if vector DB already exists
db_loc = './bn_chroma_db_mltlng_v13'
add_documents = not os.path.exists(db_loc)

# Only extract and split text if embeddings don't already exist
if add_documents:
    def lazy_load_pdf_pages(file_path):
        doc = fitz.open(file_path)
        for page_number in range(len(doc)):
            yield doc.load_page(page_number)
        doc.close()

    texts = []
    metadatas = []

    for page_number, page in enumerate(lazy_load_pdf_pages("HSC26-Bangla1st-Paper.pdf"), start=1):
        text = page.get_text("text")
        print("page_number", page_number)
        print(text)

        if text:
            texts.append(text)
            metadatas.append({"page": page_number})

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=140,
        separators=["\n", ".", "|", "!", "?", " ", ""]
    )

    documents = text_splitter.create_documents(texts, metadatas=metadatas)
    ids = [str(i) for i in range(len(documents))]

# Define embedding class
class PrefixedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")

embedding = PrefixedHuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# Initialize vector store
vect_store = Chroma(
    collection_name="my_collection",
    persist_directory=db_loc,
    embedding_function=embedding
)

# Add documents only if not already present
if add_documents:
    vect_store.add_documents(documents=documents, ids=ids)

stored = vect_store.get(include=["documents"])
print("Stored docs:", len(stored["documents"]))

# Retriever
retriever = vect_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Streamlit UI
question = st.text_input('Enter text')
if question.strip():
    result = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in result)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Keep your answers short and precise.
        Answer ONLY from the provided context.
        Answer in "Bangla" if question is in Bangla and answer in "English" if question is in English.
        If the context is insufficient, just say you don't know. 

        {context}
        Question: {question}
        """,
        input_variables = ['context', 'question']
    )

    load_dotenv()

    llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv('OPENROUTER_API_KEY'),
        model="qwen/qwen3-235b-a22b-07-25:free"
    )

    final_prompt = prompt.invoke({"context": context_text, "question": question})
    result = llm.invoke(final_prompt)

    st.markdown("## Output:")
    st.write(result.content)