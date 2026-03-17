"""Evaluation module — metrics for RAG quality assessment."""
import logging

logger = logging.getLogger(__name__)


def compute_retrieval_precision(retrieved_docs, relevant_docs):
    """Compute precision: fraction of retrieved docs that are relevant."""
    if not retrieved_docs:
        return 0.0
    retrieved_sources = {doc.get("source") for doc in retrieved_docs}
    relevant_sources = {doc.get("source") for doc in relevant_docs}
    correct = retrieved_sources & relevant_sources
    return len(correct) / len(retrieved_sources)


def compute_retrieval_recall(retrieved_docs, relevant_docs):
    """Compute recall: fraction of relevant docs that were retrieved."""
    if not relevant_docs:
        return 0.0
    retrieved_sources = {doc.get("source") for doc in retrieved_docs}
    relevant_sources = {doc.get("source") for doc in relevant_docs}
    correct = retrieved_sources & relevant_sources
    return len(correct) / len(relevant_sources)


def compute_average_score(retrieved_docs):
    """Compute mean relevance score of retrieved documents."""
    scores = [doc.get("score", 0) for doc in retrieved_docs]
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_response(question, answer, retrieved_docs, expected_keywords=None):
    """Evaluate a RAG response quality."""
    result = {
        "question": question,
        "num_sources": len(retrieved_docs),
        "avg_score": compute_average_score(retrieved_docs),
        "answer_length": len(answer),
        "has_answer": answer and "don't have enough" not in answer.lower(),
    }

    if expected_keywords:
        found = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
        result["keyword_coverage"] = found / len(expected_keywords)

    logger.info(f"Evaluation: {result}")
    return result