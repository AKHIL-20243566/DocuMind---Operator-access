"""Prompt template module — structured prompts for RAG."""


RAG_PROMPT_TEMPLATE = """You are DocuMind, an AI knowledge assistant for internal documentation.

Use ONLY the following context to answer the question.
If the context does not contain the answer, say "I don't have enough information to answer that."
Always be concise and professional.

Context:
{context}

Question:
{question}

Answer:"""


SYSTEM_PROMPT = """You are DocuMind, an AI knowledge assistant that helps employees find answers from internal company documents. You provide accurate, concise answers based solely on the retrieved context."""


def build_rag_prompt(question, context_docs):
    """Build a RAG prompt from question and retrieved documents."""
    if isinstance(context_docs, list):
        context = "\n\n".join(
            doc["text"] if isinstance(doc, dict) else str(doc)
            for doc in context_docs
        )
    else:
        context = str(context_docs)

    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


def build_standalone_prompt(question):
    """Build a prompt without context for general questions."""
    return f"""{SYSTEM_PROMPT}

Question: {question}

Answer:"""