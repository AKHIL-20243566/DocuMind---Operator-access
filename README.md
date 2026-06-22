# DocuMind — Hybrid RAG Intelligent Document Assistant

An AI-powered document intelligence platform that combines OCR, Hybrid Retrieval-Augmented Generation (RAG), and local LLM inference to answer questions from uploaded documents with high contextual accuracy.

🎥 **Demo Video:**
https://drive.google.com/file/d/1EJbwFjn6ZXuh0O6RjVIfmensLhCBliM3/view?usp=drive_link

💻 **GitHub Repository:**
[Add Repository Link]

---

# Project Overview

DocuMind was developed as part of our Semester IV Artificial Intelligence project at MNNIT.

The goal was to build an intelligent document assistant capable of understanding uploaded PDFs, scanned documents, images, and text files while generating context-aware answers grounded in the document content.

Instead of building a simple chatbot, we focused on solving a real challenge in AI systems:

> How can we retrieve the right information accurately from large unstructured documents?

To address this problem, we designed a Hybrid RAG architecture that combines semantic search, keyword retrieval, OCR-based document understanding, reranking, and local LLM generation.

---

# The Problem

Traditional document search systems generally rely on:

### Keyword Search

* Fast and simple
* Fails when wording changes
* Poor semantic understanding

### Pure Semantic Retrieval

* Understands context
* Often misses exact technical terms
* Struggles with IDs, numerical values, and structured data

This frequently results in inaccurate retrieval and lower answer quality.

---

# Our Solution

DocuMind uses a multi-stage Hybrid Retrieval Pipeline:

```text
Upload Document
        ↓
OCR / Text Extraction
        ↓
Chunking & Preprocessing
        ↓
PageIndex (Vectorless Retrieval)
        ↓
FAISS Semantic Retrieval
        ↓
BM25 Keyword Retrieval
        ↓
Reciprocal Rank Fusion (RRF)
        ↓
Cross-Encoder Reranking
        ↓
Llama 3.2B (Ollama)
        ↓
Grounded Response
```

This architecture improves retrieval accuracy by combining exact-match retrieval with semantic understanding.

---

# Core Features

### Document Understanding

* PDF support
* DOCX support
* TXT, CSV and Markdown support
* Image document support
* OCR for scanned documents

### Hybrid Retrieval

* Vector retrieval using FAISS HNSW
* BM25 keyword retrieval
* Vectorless PageIndex navigation
* Reciprocal Rank Fusion (RRF)
* Cross-encoder reranking

### AI Answer Generation

* Llama 3.2B via Ollama
* Context-aware prompting
* Grounded responses
* Streaming generation

### User Experience

* Interactive chat interface
* Retrieval confidence visualization
* Source tracking
* Context preview
* Light and Dark themes

### Security

* Institutional email authentication
* Rate limiting
* Prompt injection protection
* Query audit logging

---

# Technologies Used

### AI & Retrieval

* FAISS
* BM25
* Sentence Transformers
* Cross Encoder Reranking
* Ollama
* Llama 3.2B

### OCR & Document Processing

* PaddleOCR
* EasyOCR
* Tesseract OCR
* pdf2image

### Backend

* Python
* FastAPI

### Frontend

* React
* Vite

### DevOps

* Docker
* Docker Compose

---

# What We Learned

This project provided practical experience with:

### Retrieval-Augmented Generation

* Multi-stage retrieval pipelines
* Context construction
* Retrieval evaluation

### OCR & Document Intelligence

* Scanned document processing
* OCR optimization
* Text extraction workflows

### Information Retrieval

* FAISS indexing
* BM25 search
* Hybrid retrieval systems
* Reranking strategies

### LLM Engineering

* Prompt engineering
* Context injection
* Local model deployment with Ollama

### Full-Stack Development

* React frontend development
* FastAPI backend development
* API integration
* Authentication systems

One of the biggest lessons was realizing that retrieval quality often has a greater impact on answer quality than the language model itself.

---

# Challenges Faced

### Retrieval Accuracy

Semantic retrieval alone was not sufficient for exact queries.

**Solution**

* Added BM25 retrieval
* Implemented PageIndex
* Added retrieval fusion and reranking

### OCR Reliability

Scanned documents often produced noisy text.

**Solution**

* Implemented OCR fallback cascade
* Added preprocessing and caching

### Context Loss

Poor chunking occasionally reduced retrieval quality.

**Solution**

* Structure-aware chunking
* Overlapping context windows

### System Performance

Multiple retrieval stages increased latency.

**Solution**

* Embedding caching
* OCR caching
* Retrieval pre-filtering

---

# Team & Collaboration

This project was developed collaboratively by:

* Akhil Prakash
* Aaron Jacob
* Anirudh Sheena Vidyadharan
* Ashwin Kurian Jacob
* Aditya Patnaik

Each member contributed across development, debugging, testing, retrieval optimization, UI improvements, and system integration.

A major part of the project involved experimentation, architectural redesign, and continuous refinement of retrieval quality.

---

# Future Improvements

Planned enhancements include:

* Multi-user support
* Persistent conversation memory
* Advanced RAG agents
* Better OCR models
* Table-aware document understanding
* Multi-document reasoning
* Cloud deployment
* GPU-accelerated retrieval
* Fine-tuned domain-specific LLMs

---

# Resources

🎥 Demo Video:
https://drive.google.com/file/d/1EJbwFjn6ZXuh0O6RjVIfmensLhCBliM3/view?usp=drive_link

💻 GitHub Repository:
[Add Repository Link]

---

If you're interested in AI-powered document intelligence, retrieval systems, OCR workflows, or Hybrid RAG architectures, feel free to explore the repository and demo.
