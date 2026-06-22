# DocuMind — Hybrid RAG Intelligent Document Assistant

An AI-powered document intelligence platform that combines OCR, Hybrid Retrieval-Augmented Generation (RAG), and Local LLMs to provide accurate, context-aware answers from uploaded documents.

🎥 **Project Demo Video**
https://drive.google.com/file/d/1EJbwFjn6ZXuh0O6RjVIfmensLhCBliM3/view?usp=drive_link

---

# Overview

DocuMind was developed as part of our Semester IV Artificial Intelligence project at Motilal Nehru National Institute of Technology (MNNIT).

The project began as an academic requirement but evolved into a practical exploration of how modern AI retrieval systems are built and optimized.

Our objective was to create an intelligent document assistant capable of:

* Understanding PDFs, images, and scanned documents
* Retrieving information accurately
* Generating grounded responses using local LLMs
* Improving retrieval quality through hybrid search techniques

Rather than building a simple chatbot, we focused on solving a real-world AI challenge:

> How can we retrieve the right information from large unstructured documents while maintaining both contextual understanding and factual accuracy?

---

# The Problem

Traditional document search systems generally rely on one of two approaches:

### Keyword-Based Search

* Fast and efficient
* Works well for exact matches
* Struggles with contextual understanding

### Semantic Search

* Understands meaning and context
* Handles natural language queries
* Can miss exact technical terms, IDs, definitions, and structured information

Both approaches have strengths and weaknesses.

To overcome these limitations, we designed a Hybrid Retrieval Architecture that combines semantic understanding with exact-match retrieval.

---

# Our Solution

DocuMind combines multiple retrieval strategies into a single intelligent pipeline.

```text
Document Upload
      ↓
Document Parsing / OCR
      ↓
Text Cleaning & Chunking
      ↓
PageIndex (Vectorless Navigation)
      ↓
FAISS Semantic Retrieval
      ↓
BM25 Keyword Retrieval
      ↓
Reciprocal Rank Fusion (RRF)
      ↓
Cross Encoder Reranking
      ↓
Llama 3.2B (Ollama)
      ↓
Grounded AI Response
```

This architecture significantly improves retrieval quality compared to traditional RAG implementations.

---

# Key Features

## Document Understanding

* PDF support
* DOCX support
* TXT support
* CSV support
* Markdown support
* Image support
* Scanned document processing

## OCR Processing

During development we evaluated:

* Tesseract OCR
* EasyOCR
* PaddleOCR

After extensive testing, PaddleOCR was selected as the final OCR solution due to its higher accuracy and better performance on scanned academic and structured documents.

Final OCR workflow:

```text
Scanned PDF / Image
        ↓
Image Preprocessing
        ↓
PaddleOCR
        ↓
Text Cleaning
        ↓
Chunking
        ↓
Indexing
```

## Hybrid Retrieval System

* FAISS HNSW vector search
* BM25 keyword retrieval
* PageIndex vectorless navigation
* Reciprocal Rank Fusion (RRF)
* Cross Encoder reranking

## AI Answer Generation

* Llama 3.2B via Ollama
* Context-aware prompting
* Grounded answer generation
* Streaming responses

## User Experience

* Interactive AI chat interface
* Retrieval confidence visualization
* Context preview system
* Source tracking
* Responsive UI
* Light and Dark themes

## Security Features

* Institutional email authentication
* API protection
* Prompt injection defenses
* Rate limiting
* Query audit logging

---

# Technologies Used

## AI & Retrieval

* FAISS (HNSW)
* BM25
* Sentence Transformers
* Cross Encoder Reranking
* Ollama
* Llama 3.2B

## OCR & Document Processing

* PaddleOCR
* pdf2image
* Poppler

## Backend

* Python
* FastAPI

## Frontend

* React
* Vite
* Lucide React

## Infrastructure

* Docker
* Docker Compose

---

# What We Learned

This project provided practical exposure to several important AI engineering concepts.

### Retrieval-Augmented Generation (RAG)

* Multi-stage retrieval pipelines
* Context construction
* Grounded generation
* Retrieval evaluation

### Information Retrieval

* FAISS indexing
* Hybrid search systems
* BM25 retrieval
* Retrieval fusion
* Cross-encoder reranking

### OCR & Document Intelligence

* Scanned document processing
* OCR optimization
* Text extraction workflows
* Data cleaning pipelines

### LLM Engineering

* Local LLM deployment
* Prompt engineering
* Context injection
* Streaming generation

### Full Stack Development

* React frontend architecture
* FastAPI backend APIs
* Authentication systems
* API integration
* UI/UX design

One of our biggest takeaways was realizing that improving retrieval quality often has a larger impact on answer quality than changing the language model itself.

---

# Challenges Faced

## Retrieval Accuracy

Initially, semantic retrieval alone was not sufficient for exact queries.

### Solution

* Added BM25 retrieval
* Introduced PageIndex
* Implemented retrieval fusion
* Added reranking

---

## OCR Reliability

Early OCR experiments produced inconsistent results on scanned documents.

### Solution

* Evaluated multiple OCR frameworks
* Migrated to PaddleOCR
* Added image preprocessing
* Improved extraction quality significantly

---

## Context Loss

Chunk boundaries occasionally caused loss of important context.

### Solution

* Structure-aware chunking
* Chunk overlap strategies
* Section-based retrieval

---

## Performance Optimization

The multi-stage retrieval pipeline increased computational overhead.

### Solution

* Embedding caching
* OCR caching
* Retrieval pre-filtering
* Efficient indexing strategies

---

# Team & Collaboration

This project was developed collaboratively by:

* Akhil Prakash
* Aaron Jacob
* Anirudh Sheena Vidyadharan
* Ashwin Kurian Jacob
* Aditya Patnaik

The project involved continuous experimentation, debugging, testing, and optimization across both AI and software engineering components.

Each team member contributed to various aspects of the project, including retrieval systems, OCR integration, frontend development, backend APIs, system architecture, testing, and overall project refinement.

---

# Real-World Applications

DocuMind can be extended to:

* Enterprise knowledge assistants
* Academic research assistants
* Legal document retrieval systems
* Healthcare documentation systems
* Internal organizational search engines
* AI-powered knowledge management platforms

---

# Future Improvements

Planned enhancements include:

* Multi-user support
* Persistent chat history
* Role-based access control
* Better document visualization
* Table-aware document understanding
* Multi-document reasoning
* Cloud deployment
* GPU-accelerated inference
* Advanced Agentic RAG workflows
* Fine-tuned domain-specific models

---

# Project Resources

🎥 Demo Video
https://drive.google.com/file/d/1EJbwFjn6ZXuh0O6RjVIfmensLhCBliM3/view?usp=drive_link

💻 GitHub Repository
[Add Repository Link]

If you're interested in Retrieval-Augmented Generation, Document Intelligence, OCR systems, Information Retrieval, or Local LLM applications, feel free to explore the repository and demo.

---

## License

MIT License
