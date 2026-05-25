# Explainable Multi-Document RAG System

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDF documents, ask natural language questions, and receive grounded answers with source tracking and PDF highlighting.

The system combines semantic search, vector databases, and local LLM inference to build an explainable document assistant.

---

## Features

- Upload and query multiple PDFs
- Semantic search using embeddings
- Local LLM inference using Ollama
- Source tracking with page and chunk references
- Similarity score display
- PDF answer highlighting
- Chat history
- Persistent ChromaDB storage

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF |
| Chunking | LangChain |
| Embeddings | sentence-transformers |
| Vector Database | ChromaDB |
| LLM | Ollama (llama3.2:1b) |
| Language | Python |

---

## Architecture

```text
PDF Upload
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Similarity Search
    ↓
LLM (Ollama)
    ↓
Answer Generation
    ↓
Source Tracking + PDF Highlighting
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd <repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Download Ollama:

https://ollama.com/download

Pull the model:

```bash
ollama pull llama3.2:1b
```

### 4. Run the application

```bash
streamlit run streamlit_app.py
```

---

## Project Structure

```text
project/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
├── uploaded_docs/
├── chroma_dbs/
└── screenshots/
```

---

## How It Works

1. PDFs are uploaded through Streamlit
2. Text is extracted page-by-page using PyMuPDF
3. Documents are split into overlapping chunks
4. Chunks are converted into embeddings
5. Embeddings are stored in ChromaDB
6. User queries are embedded and matched using similarity search
7. Retrieved chunks are passed to the LLM
8. The system generates grounded answers with source references

---

## Key Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Databases
- Embeddings
- Prompt Engineering
- Explainable AI
- Local LLM Deployment
- Source Grounding

---

## Future Improvements

- Cloud deployment
- OCR support for scanned PDFs
- Hybrid search
- Reranking
- Conversational memory
- Advanced highlighting UI

---

## Author

Medhini Sasanapuri

