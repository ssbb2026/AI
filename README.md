---
title: AI
emoji: 💻
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.28.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: AI
---
# AI Document Intelligence & Retrieval-Augmented Generation (RAG)

## 📌 Project Overview

PROJECT LINK:https://huggingface.co/spaces/ssbb2026/AI

An AI-powered **Document Intelligence and Retrieval-Augmented Generation (RAG)** application that enables users to ask natural-language questions about PDF documents.

The system extracts content from a PDF, converts it into structured Markdown, creates structure-aware chunks, generates semantic embeddings using **Sentence Transformers**, and stores them in a **FAISS vector index** for efficient similarity search.

When a user asks a question, the system retrieves the most relevant document chunks and provides them as context to a **Qwen Large Language Model**, which generates a document-grounded answer.

The application is presented through an interactive **Gradio chatbot interface**.

---

## 🔄 RAG Pipeline

```text
PDF Document
     ↓
PyMuPDF4LLM
     ↓
Structured Markdown
     ↓
Structure-Aware Chunking
     ↓
Sentence Transformer Embeddings
     ↓
FAISS Vector Index
     ↓
Semantic Retrieval
     ↓
Relevant Document Context
     ↓
Qwen LLM
     ↓
Grounded Answer
     ↓
Gradio Chatbot
```

---

## 🎯 Key Features

* PDF-to-Markdown document extraction
* Structure-aware document chunking
* Chapter, section, and subsection metadata
* Semantic text embeddings
* FAISS vector similarity search
* Retrieval-Augmented Generation
* Document-grounded question answering
* Source chunk and similarity-score display
* CPU and CUDA/GPU support
* Interactive Gradio chatbot
* Retrieval evaluation framework

---

## 📊 RAG Evaluation

The project includes retrieval evaluation using:

* **Hit@K**
* **Precision@K**
* **Recall@K**
* **Mean Reciprocal Rank (MRR)**

A generation-testing framework is also included to compare:

```text
Ground Truth Answer
        vs.
Generated Answer
```

while retaining the retrieved context and source information for analysis.

---

## 🛠️ Technologies

* **Python**
* **PyMuPDF4LLM**
* **Sentence Transformers**
* **FAISS**
* **Hugging Face Transformers**
* **Qwen/Qwen2.5-0.5B-Instruct**
* **PyTorch**
* **NumPy**
* **Gradio**

---

## 🧠 AI/ML Concepts

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation (RAG)
* Generative AI
* Large Language Models (LLMs)
* Natural Language Processing
* Semantic Search
* Text Embeddings
* Vector Search
* Document Intelligence
* Prompt Engineering
* Context Retrieval
* Information Retrieval Evaluation

---

## ▶️ Run the Project

Install the required dependencies:

```bash
pip install pymupdf4llm
pip install sentence-transformers
pip install faiss-cpu
pip install transformers
pip install accelerate
pip install gradio
pip install torch
```

Place the PDF in the project directory:

```text
Earth Our Planet data from web.pdf
```

Run the application:

```bash
python app.py
```

The Gradio interface will launch and allow users to ask questions about the document.

---

## 🚀 Future Enhancements

* Multi-document RAG
* Document upload through the UI
* Reranking models
* Hybrid keyword + semantic search
* OCR support for scanned PDFs
* RAGAS-based evaluation
* Answer-level evaluation metrics
* Persistent FAISS indexes
* Document citations
* Conversation memory
* Docker deployment
* Cloud deployment

---

## 👨‍💻 Project Focus

This project demonstrates an end-to-end **Generative AI and Document Intelligence workflow**, combining document processing, semantic retrieval, vector search, and LLM-based answer generation into a single RAG application.
