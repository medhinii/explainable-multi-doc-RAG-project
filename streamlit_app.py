import os
import re
import hashlib
import tempfile
import fitz
import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM


# ============================================
# PATHS
# ============================================

BASE_DIR = r"C:\Users\sasan\OneDrive\Documents\rag project"

UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_docs")
DB_DIR = os.path.join(BASE_DIR, "chroma_dbs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)


# ============================================
# SESSION STATE
# ============================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================
# TEXT CLEANING
# ============================================

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"(?<=[a-z])(?=[A-Z])",
        " ",
        text
    )

    return text.strip()


# ============================================
# HASHING
# ============================================

def file_hash(file_path):

    hasher = hashlib.md5()

    with open(file_path, "rb") as f:
        hasher.update(f.read())

    return hasher.hexdigest()


def multiple_file_hash(file_paths):

    combined = ""

    for path in sorted(file_paths):
        combined += file_hash(path)

    return hashlib.md5(combined.encode()).hexdigest()


# ============================================
# LOAD + SPLIT MULTIPLE PDFs
# ============================================

def load_and_split_multiple_pdfs(pdf_paths):

    all_documents = []

    for pdf_path in pdf_paths:

        pdf = fitz.open(pdf_path)

        for page_num in range(len(pdf)):

            page = pdf[page_num]

            blocks = page.get_text("blocks")

            text = "\n".join([
                block[4]
                for block in blocks
            ])

            text = clean_text(text)

            if text.strip() == "":
                continue

            # skip references section
            if (
                "references" in text.lower()
                and page_num > len(pdf) * 0.7
            ):
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(pdf_path),
                    "file_path": pdf_path,
                    "page": page_num + 1
                }
            )

            all_documents.append(doc)

        pdf.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(all_documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i + 1

    return chunks


# ============================================
# EMBEDDINGS
# ============================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================
# LLM
# ============================================

@st.cache_resource
def get_llm():

    return OllamaLLM(
        model="llama3.2:1b"
    )


# ============================================
# VECTORSTORE
# ============================================

def create_or_load_vectorstore(pdf_paths):

    embeddings = get_embeddings()

    collection_id = multiple_file_hash(pdf_paths)

    db_path = os.path.join(
        DB_DIR,
        collection_id
    )

    if os.path.exists(db_path):

        st.info("Loading existing vector database...")

        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )

        chunks_count = "Already processed"

    else:

        st.info("Processing PDFs for the first time...")

        chunks = load_and_split_multiple_pdfs(
            pdf_paths
        )

        st.info(
            f"Creating embeddings for {len(chunks)} chunks..."
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )

        chunks_count = len(chunks)

    return vectorstore, chunks_count


# ============================================
# ANSWER GENERATION
# ============================================

def generate_answer(vectorstore, query):

    llm = get_llm()

    results_with_scores = (
        vectorstore.similarity_search_with_score(
            query,
            k=3
        )
    )

    results = [
        doc
        for doc, score in results_with_scores
    ]

    context = "\n\n".join([

        f"""
Source File: {doc.metadata.get('source', 'N/A')}
Page: {doc.metadata.get('page', 'N/A')}
Chunk: {doc.metadata.get('chunk_id', 'N/A')}

{doc.page_content}
"""

        for doc in results

    ])

    prompt = f"""
You are a document question-answering assistant.

Use ONLY the context below.

Give a clear answer in 3-5 simple sentences.

Mention source file names and page numbers when useful.

If answer is not found, say:
"I could not find this in the uploaded documents."

Context:
{context}

Question:
{query}

Answer:
"""

    answer = llm.invoke(prompt)

    return answer, results_with_scores


# ============================================
# PDF HIGHLIGHTING
# ============================================

def highlight_pdf(
    pdf_path,
    text_to_highlight,
    page_number
):

    pdf = fitz.open(pdf_path)

    page = pdf[page_number - 1]

    search_text = text_to_highlight[:120]

    matches = page.search_for(search_text)

    for match in matches:

        highlight = page.add_highlight_annot(
            match
        )

        highlight.update()

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    output_path = temp_file.name

    temp_file.close()

    pdf.save(output_path)

    pdf.close()

    return output_path


# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Assistant")

st.write(
    """
Upload one or more PDFs,
ask questions,
and get grounded answers
with source tracking.
"""
)

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("Project Info")

st.sidebar.write(
    "**Project:** Multi-PDF RAG System"
)

st.sidebar.write(
    "**Vector DB:** ChromaDB"
)

st.sidebar.write(
    "**Embeddings:** MiniLM"
)

st.sidebar.write(
    "**LLM:** Ollama llama3.2:1b"
)

st.sidebar.write(
    "**Features:**"
)

st.sidebar.write(
    "- Multiple PDF support"
)

st.sidebar.write(
    "- Source tracking"
)

st.sidebar.write(
    "- Similarity scores"
)

st.sidebar.write(
    "- PDF highlighting"
)

st.sidebar.write(
    "- Chat history"
)


# ============================================
# FILE UPLOAD
# ============================================

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================
# MAIN LOGIC
# ============================================

if uploaded_files:

    pdf_paths = []

    # save uploaded PDFs
    for uploaded_file in uploaded_files:

        pdf_path = os.path.join(
            UPLOAD_DIR,
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        pdf_paths.append(pdf_path)

    st.success(
        f"{len(uploaded_files)} PDF(s) uploaded successfully!"
    )

    # vector DB
    with st.spinner(
        "Loading or creating vector database..."
    ):

        vectorstore, chunks_count = (
            create_or_load_vectorstore(
                pdf_paths
            )
        )

    st.session_state.vectorstore = vectorstore

    st.success(
        f"Documents ready! Chunks: {chunks_count}"
    )

    # uploaded file names
    st.subheader("Uploaded Documents")

    for file in uploaded_files:
        st.write(f"📄 {file.name}")

    # question input
    query = st.text_input(
        "Ask a question from the uploaded documents"
    )

    # ========================================
    # QUESTION ANSWERING
    # ========================================

    if query:

        with st.spinner("Generating answer..."):

            answer, results = generate_answer(
                st.session_state.vectorstore,
                query
            )

        # save history
        st.session_state.chat_history.append(
            {
                "question": query,
                "answer": answer
            }
        )

        # ====================================
        # ANSWER
        # ====================================

        st.subheader("Answer")

        st.write(answer)

        # ====================================
        # SOURCES
        # ====================================

        st.subheader("Sources Used")

        for i, (doc, score) in enumerate(results, 1):

            source_file = doc.metadata.get(
                "source",
                "N/A"
            )

            file_path = doc.metadata.get(
                "file_path",
                "N/A"
            )

            page = doc.metadata.get(
                "page",
                "N/A"
            )

            chunk_id = doc.metadata.get(
                "chunk_id",
                "N/A"
            )

            st.markdown(
                f"### Source {i}"
            )

            st.write(
                f"**File:** {source_file}"
            )

            st.write(
                f"**Page Number:** {page}"
            )

            st.write(
                f"**Chunk ID:** {chunk_id}"
            )

            st.write(
                f"**Similarity Score:** {score:.4f}"
            )

            # source preview
            with st.expander(
                "View source text"
            ):

                st.write(
                    doc.page_content
                )

            # =================================
            # PDF HIGHLIGHT DOWNLOAD
            # =================================

            try:

                highlighted_pdf = highlight_pdf(
                    file_path,
                    doc.page_content,
                    page
                )

                with open(
                    highlighted_pdf,
                    "rb"
                ) as f:

                    st.download_button(
                        label=(
                            f"Download Highlighted PDF "
                            f"- Source {i}"
                        ),
                        data=f,
                        file_name=(
                            f"highlighted_{source_file}"
                        ),
                        mime="application/pdf"
                    )

            except Exception:

                st.warning(
                    """
Could not highlight this source,
but source file/page/chunk
are still shown above.
"""
                )

    # ========================================
    # CHAT HISTORY
    # ========================================

    st.subheader("Chat History")

    for chat in reversed(
        st.session_state.chat_history
    ):

        st.markdown(
            f"**Question:** {chat['question']}"
        )

        st.markdown(
            f"**Answer:** {chat['answer']}"
        )

        st.markdown("---")

else:

    st.info(
        "Please upload one or more PDFs to begin."
    )