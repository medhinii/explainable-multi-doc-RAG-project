import os
import re
import hashlib
import fitz
import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM


BASE_DIR = r"C:\Users\sasan\OneDrive\Documents\rag project"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_docs")
DB_DIR = os.path.join(BASE_DIR, "chroma_dbs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return text.strip()


def file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_and_split(pdf_path):
    pdf = fitz.open(pdf_path)
    documents = []

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text("blocks")
        text = "\n".join([block[4] for block in text])
        text = clean_text(text)

        if text.strip() == "":
            continue

        if "references" in text.lower() and page_num > len(pdf) * 0.7:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "page": page_num + 1
                }
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i + 1

    return chunks


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_or_load_vectorstore(pdf_path):
    embeddings = get_embeddings()

    pdf_id = file_hash(pdf_path)
    db_path = os.path.join(DB_DIR, pdf_id)

    if os.path.exists(db_path):
        st.info("Loading existing vector database...")
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        chunks_count = "Already processed"

    else:
        st.info("Processing PDF for the first time...")
        chunks = load_and_split(pdf_path)

        st.info(f"Creating embeddings for {len(chunks)} chunks...")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )

        chunks_count = len(chunks)

    return vectorstore, chunks_count


def generate_answer(vectorstore, query):
    llm = OllamaLLM(model="llama3.2:1b")

    results_with_scores = vectorstore.similarity_search_with_score(
        query,
        k=3
    )

    results = [doc for doc, score in results_with_scores]

    context = "\n\n".join([
        f"Source: Page {doc.metadata.get('page', 'N/A')}, "
        f"Chunk {doc.metadata.get('chunk_id', 'N/A')}\n{doc.page_content}"
        for doc in results
    ])

    prompt = f"""
You are a document question-answering assistant.

Use ONLY the context below.
Give a clear answer in 3-5 simple sentences.
Do NOT just repeat the question or title.
If the answer is not present, say:
"I could not find this in the document."

Context:
{context}

Question:
{query}

Answer:
"""

    answer = llm.invoke(prompt)

    return answer, results_with_scores


st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Smart RAG Document Assistant")
st.write("Upload a PDF, ask a question, and get an answer with source page, chunk, and similarity score.")

st.sidebar.title("Project Info")
st.sidebar.write("**Project:** RAG Document Q&A")
st.sidebar.write("**Vector DB:** ChromaDB")
st.sidebar.write("**Embeddings:** MiniLM")
st.sidebar.write("**LLM:** Ollama llama3.2:1b")
st.sidebar.write("**Feature:** Source tracking")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    with st.spinner("Loading or creating vector database..."):
        vectorstore, chunks_count = create_or_load_vectorstore(pdf_path)

    st.session_state.vectorstore = vectorstore
    st.success(f"Document ready! Chunks: {chunks_count}")

    query = st.text_input("Ask a question from the document")

    if query:
        with st.spinner("Generating answer..."):
            answer, results = generate_answer(
                st.session_state.vectorstore,
                query
            )

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources Used")

        for i, (doc, score) in enumerate(results, 1):
            page = doc.metadata.get("page", "N/A")
            chunk_id = doc.metadata.get("chunk_id", "N/A")

            st.markdown(f"### Source {i}")
            st.write(f"**Page Number:** {page}")
            st.write(f"**Chunk ID:** {chunk_id}")
            st.write(f"**Similarity Score:** {score:.4f}")

            with st.expander("View source text"):
                st.write(doc.page_content)

else:
    st.info("Please upload a PDF to begin.")