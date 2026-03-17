"""FastAPI routes for the DocuMind RAG API."""
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocuMind RAG API",
    description="AI Knowledge Assistant — RAG-powered document Q&A",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


@app.on_event("startup")
async def startup():
    pipeline = RAGPipeline.get_instance()
    pipeline.initialize()
    logger.info("RAG pipeline ready")


@app.get("/health")
def health():
    return {"status": "ok", "message": "DocuMind RAG API is running"}


@app.post("/chat")
def chat(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        pipeline = RAGPipeline.get_instance()
        result = pipeline.query(req.question)
        logger.info(f"Query: '{req.question[:50]}' → {len(result['sources'])} sources")
        return result
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))