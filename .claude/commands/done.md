## 🚨 Core System Rules (DO NOT VIOLATE)

* The system MUST use **Vectorless RAG (PageIndex-style reasoning)** as the primary retrieval method
* DO NOT rely on vector similarity search as the main retrieval mechanism
* DO NOT use naive or fixed-size chunking
* Documents must be structured into:

  * Pages
  * Sections
  * Subsections
* Retrieval MUST follow hierarchical reasoning:

  * Start broad → narrow down → validate → refine
* Context passed to LLM must be:

  * Minimal
  * Highly relevant
  * Structured

### Retrieval Behavior

* Simulate human-like search (step-by-step traversal)
* Support iterative retrieval (refine if answer not found)
* Handle references like “see section X” properly
* Use reasoning, not similarity

### Answer Generation Rules

* Answer ONLY from retrieved context
* If answer not found → say: "Not found in document"
* No hallucination

---

## ⚙️ Performance + Code Quality Rules

* Remove unused files, dead code, and redundant dependencies
* Avoid unnecessary API calls and duplicate processing
* Optimize for fast response time and low memory usage
* Keep architecture modular:

  * /api
  * /rag
  * /llm
  * /utils

---

## 🧪 Stability Rules

* System must not crash on:

  * Empty input
  * Invalid queries
  * Missing documents
* Always return a safe fallback response

---

## 🔐 MANDATORY PRE-COMPLETION CHECKLIST

Run the mandatory pre-completion checklist on all work done in this conversation before we call the task complete. Work through each step carefully:

---

// (KEEP YOUR EXISTING CHECKLIST BELOW EXACTLY SAME — DO NOT MODIFY)
